import numpy as np
import torch

from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.agents.common.seed import set_seed_except_env_seed
from enlighten.agents.common.other import get_device
from enlighten.envs.multi_nav_env import MultiNavEnv
from enlighten.agents.common.other import get_obs_channel_num
from enlighten.datasets.il_data_gen import load_behavior_dataset_meta, extract_observation
from enlighten.agents.models.decision_transformer import DecisionTransformer
from enlighten.agents.models.rnn_seq_model import RNNSequenceModel
from enlighten.agents.evaluation.across_scene_base_evaluator import AcrossEnvBaseEvaluator
from enlighten.envs.vec_env import construct_envs_based_on_dataset
from enlighten.envs import VectorEnv
from enlighten.envs.nav_env import NavEnv

from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)
from enlighten.utils.ddp_utils import (
    EXIT,
    add_signal_handlers,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)

from torch import Size, Tensor
import tqdm

# evaluate an agent across scene vector envs
class AcrossEnvEvaluatorVector(AcrossEnvBaseEvaluator):
    @staticmethod
    def _pause_envs(
        envs_to_pause: List[int],
        envs: Union[VectorEnv, NavEnv],
        recurrent_hidden_states: Tensor,
        not_done_masks: Tensor,
        current_episode_reward: Tensor,
        prev_actions: Tensor,
        observations: Tensor,
        goals: Tensor
    ) -> Tuple[
        Union[VectorEnv, NavEnv],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor
    ]:
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # only keep the indices of non-paused envs, i.e. state_index (a list of indices)
            # indexing along the batch dimensions
            recurrent_hidden_states = recurrent_hidden_states[:,state_index]
            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]
            observations = observations[state_index]
            goals = goals[state_index]

        return (
            envs,
            recurrent_hidden_states,
            not_done_masks,
            current_episode_reward,
            prev_actions,
            observations,
            goals
        )

    def get_number_of_eval_episodes(self):
        # sum over all envs
        number_of_eval_episodes = sum(self.envs.number_of_episodes)
        

        print("===> Number of eval environments: %d"%(self.config.get("num_environments")))
        print("===> Number of eval episodes: %d"%(number_of_eval_episodes))        
    
        return number_of_eval_episodes

    # extract scalars (success, spl) from info
    # scalars are transformed to float
    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            if np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    # original_observations: a list of observations returned from B vector envs
    # observations: [B,C,H,W]
    # goals: [B,goal_dim]
    def observation2agentinput(self, original_observations, goal_form):
        # goals: float list --> (B,1)
        goals = []
        obs_arrays = []
        if goal_form == "distance_to_goal":
            goals = self.envs.get_distance_to_goal()
            goals = torch.from_numpy(np.concatenate(goals, axis=0)).to(dtype=torch.float32, device=self.device).reshape(-1, 1)

        for i, obs in enumerate(original_observations):
            # o: (C,H,W) --> (1,C,H,W)
            obs_array = extract_observation(obs, self.envs.observation_spaces[i].spaces)
            obs_array = np.expand_dims(obs_array, axis=0)
            obs_arrays.append(obs_array)
            # g: (goal_dim,) --> (1,goal_dim)
            if goal_form == "rel_goal":
                goal = np.array(obs["pointgoal"], dtype="float32")
                goal = np.expand_dims(goal, axis=0)
                goals.append(goal)
           
        # observations: (B,C,H,W)
        observations = torch.from_numpy(np.concatenate(obs_arrays, axis=0)).to(device=self.device, dtype=torch.float32)
        # goals: (B,goal_dim)
        if goal_form == "rel_goal":
            goals = torch.from_numpy(np.concatenate(goals, axis=0)).to(device=self.device, dtype=torch.float32) 

        return observations, goals


    def evaluate_over_one_dataset_rnn(self, model, sample, split_name, logs):
        # initialize data structures of stats
        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode, episode id is a combination of episode id and scene id

        
        # get number of eval episodes
        number_of_eval_episodes = self.get_number_of_eval_episodes()
        
        # get tqdm bar
        pbar = tqdm.tqdm(total=number_of_eval_episodes)

        # reset envs and get initial observations
        original_observations = self.envs.reset()

        # observations: [B,C,H,W]
        # goals: [B,goal_dim]
        observations, goals = self.observation2agentinput(original_observations, self.goal_form)

        # initialize algorithm data structures
        # a0 is -1, shape [B]
        prev_actions = torch.ones((self.envs.num_envs), device=self.device, dtype=torch.long) * (-1)

        # h0 is 0, shape [1, B, hidden_size]
        h = torch.zeros(1, self.envs.num_envs, 
            int(self.config.get("rnn_hidden_size")),
            dtype=torch.float32,
            device=self.device,
        )

        # dones
        not_done_masks = torch.zeros(
            self.envs.num_envs, 1, device=self.device, dtype=torch.bool
        )
        
        # evaluate all episodes
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            # get current episodes
            current_episodes = self.envs.current_episodes()

            # print(actions.size())
            # print(observations.size())
            # print(goals.size())
            # print(h.size())

            # act agent
            actions, h = model.get_action(
                observations,
                prev_actions,
                goals,
                h,
                sample=sample)
            
            # actions: [B,1] --> [B]
            actions = torch.squeeze(actions, 1)
            
            # print(actions)
            # print(actions.size())
            # print(h.size())

            with torch.no_grad():
                prev_actions.copy_(actions)

            #print(actions)
            #exit()

            # Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to an int
            step_data = [{"action": a.item()} for a in actions.to(device="cpu")]

            # step envs
            outputs = self.envs.step(step_data)

            # unpack outputs
            original_observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            # get new observations and goals
            observations, goals = self.observation2agentinput(original_observations, self.goal_form)

            # get new done masks in cpu
            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            # get new rewards
            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)

            # compute return of current episode as a vector
            current_episode_reward += rewards

            # get next episodes (to check envs to pause)
            next_episodes = self.envs.current_episodes()

            # get envs to pause if the episode assigned to the environment has ended
            # if an env has paused, it won't be stepped 
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended (done=True)
                # only record an episode's metric when it ends
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {}
                    # record return
                    episode_stats["return"] = current_episode_reward[i].item()
                    # extract and record other evaluation metrics from infos, succeed, spl
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    # reset return as 0
                    current_episode_reward[i] = 0
                    # record stats for current episode
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

            # pause envs 
            # all returned values should be updated according to non-paused envs
            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                h,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                observations,
                goals
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                h,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                observations,
                goals
            )
        
        return self.get_metric_logs(split_name, stats_episodes, logs)
            

    def evaluate_over_one_dataset(self, model, sample, split_name, logs):
        self.envs = construct_envs_based_on_dataset(
                config=self.config,
                split_name=split_name,
                workers_ignore_signals=is_slurm_batch_job()
            )
        
        if self.algorithm_name == "rnn":
            logs = self.evaluate_over_one_dataset_rnn(model, sample, split_name, logs)
        else:
            print("Not implementd: dt vector env evaluation")
            exit()    

        # close envs
        self.envs.close()

        return logs
    
    def evaluate_over_datasets(self, model=None, sample=True):
        if model is None:
            model = self.load_model()

        logs = {}
        for split_name, _ in self.eval_dataset_episodes.items():
            logs = self.evaluate_over_one_dataset(model, sample, split_name, logs)
        
        return logs

    def get_metric_logs(self, split_name, stats_episodes, logs):
        # number of episodes
        num_episodes = len(stats_episodes)
        logs[f"{split_name}/total_episodes"] = num_episodes
        # average metrics
        for stat_key in next(iter(stats_episodes.values())).keys():
            if stat_key == "success":
                print_key = "success_rate"
            elif stat_key == "spl": 
                print_key = "mean_spl"   
            else:
                print_key = stat_key
            logs[f"{split_name}/{print_key}"] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        return logs

    

if __name__ == "__main__":
    eval_splits = ["same_start_goal_test", "same_scene_test", "across_scene_test"]
    evaluator = AcrossEnvEvaluatorVector(eval_splits=eval_splits, config_filename="imitation_learning_rnn.yaml") 
    logs = evaluator.evaluate_over_datasets(sample=True)
    evaluator.print_metrics(logs, eval_splits)
    evaluator.save_eval_logs(logs, eval_splits)