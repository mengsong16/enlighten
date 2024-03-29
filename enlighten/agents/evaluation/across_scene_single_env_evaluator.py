import numpy as np
import torch

from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.agents.common.seed import set_seed_except_env_seed
from enlighten.agents.common.other import get_device
from enlighten.envs.multi_nav_env import MultiNavEnv
from enlighten.agents.common.other import get_obs_channel_num
from enlighten.datasets.common import extract_observation
from enlighten.datasets.common import load_behavior_dataset_meta 
from enlighten.agents.models.decision_transformer import DecisionTransformer
from enlighten.agents.models.rnn_seq_model import RNNSequenceModel
from enlighten.agents.evaluation.across_scene_base_evaluator import AcrossEnvBaseEvaluator
from enlighten.agents.evaluation.ppo_eval import *
from enlighten.datasets.common import goal_position_to_abs_goal

class MeasureHistory:
    def __init__(self, id):
        self.id = id
        self.data = []
    
    def add(self, a):
        self.data.append(a)
    
    def len(self):
        return len(self.data)

    def mean(self): 
        data = np.array(self.data, dtype=np.float32) 
        return np.mean(data, axis=0)

    def max(self):
        data = np.array(self.data, dtype=np.float32)
        return np.max(data, axis=0)
    
    def min(self):
        data = np.array(self.data, dtype=np.float32)
        return np.min(data, axis=0)

    def std(self):
        data = np.array(self.data, dtype=np.float32)
        return np.std(data, axis=0)
    
    def print_full_summary(self):
        print("================  %s  ======================"%(self.id))
        print("Num: %f"%self.len())
        print("Min: %f"%self.min())
        print("Mean: %f"%self.mean())
        print("Max: %f"%self.max())
        print("Std: %f"%self.std())
        print("==============================================")



# evaluate decision transformer for one episode
def evaluate_one_episode_dt(
        episode,
        env,
        model,
        goal_form,
        sample,
        max_ep_len,
        device,
        goal_dimension, 
        goal_coord_system
    ):

    # turn model into eval mode and move to desired device
    model.eval()
    model.to(device=device)

    # reset env
    obs = env.reset(episode=episode, plan_shortest_path=False)
    obs_array = extract_observation(obs, env.observation_space.spaces)
    if goal_form == "rel_goal":
        goal = np.array(obs["pointgoal"], dtype="float32")
    elif goal_form == "distance_to_goal":
        goal = env.get_current_distance()
    elif goal_form == "abs_goal":
        goal_position = np.array(env.goal_position, dtype="float32")
        goal = goal_position_to_abs_goal(goal_position,
            goal_dimension, goal_coord_system)
    # a0 is 0, shape (1,1)
    actions = torch.zeros((1, 1), device=device, dtype=torch.long)

    # print("Scene id: %s"%(episode.scene_id))
    # print("Goal position: %s"%(env.goal_position))
    # print("Start position: %s"%(env.start_position))

    # change shape and convert to torch tensor
    # (C,H,W) --> (1,1,C,H,W)
    obs_array = np.expand_dims(np.expand_dims(obs_array, axis=0), axis=0)
    observations = torch.from_numpy(obs_array).to(device=device, dtype=torch.float32)
    # (goal_dim,) --> (1,1,goal_dim)
    if goal_form == "rel_goal" or goal_form == "abs_goal":
        goal = np.expand_dims(np.expand_dims(goal, axis=0), axis=0)
        goals = torch.from_numpy(goal).to(device=device, dtype=torch.float32)
    # float --> (1,1,1)
    elif goal_form == "distance_to_goal":
        goals = torch.tensor(goal, device=device, dtype=torch.float32).reshape(1, 1, 1)

    # t0 is 0, shape (1,1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    real_act_seqs = []
    # run under policy for max_ep_len step or done
    # keep all history steps, but only use context length to predict current action
    for t in range(max_ep_len):
        # predict according to the sequence from (s0,a0,r0) up to now (context)
        # need to input timesteps as positional embedding
        action = model.get_action(
            observations,
            actions,
            goals,
            timesteps,
            sample=sample,
        )
        
        # right append new action
        action = action.detach().cpu().item()
        real_act_seqs.append(action)

        new_action = torch.tensor(action, device=device, dtype=torch.long).reshape(1, 1)
        actions = torch.cat([actions, new_action], dim=1)

        # step the env according to the action, get new observation and goal
        obs, _, done, _ = env.step(action)
        obs_array = extract_observation(obs, env.observation_space.spaces)
        if goal_form == "rel_goal":
            goal = np.array(obs["pointgoal"], dtype="float32")
        elif goal_form == "distance_to_goal":
            goal = env.get_current_distance()
        elif goal_form == "abs_goal":
            goal_position = np.array(env.goal_position, dtype="float32")
            goal = goal_position_to_abs_goal(goal_position,
                goal_dimension, goal_coord_system)

        # change shape and convert to torch tensor
        # (C,H,W) --> (1,1,C,H,W)
        obs_array = np.expand_dims(np.expand_dims(obs_array, axis=0), axis=0)
        new_obs = torch.from_numpy(obs_array).to(device=device, dtype=torch.float32)
        # (goal_dim,) --> (1,1,goal_dim)
        if goal_form == "rel_goal" or goal_form == "abs_goal":
            goal = np.expand_dims(np.expand_dims(goal, axis=0), axis=0)
            new_goal = torch.from_numpy(goal).to(device=device, dtype=torch.float32)
        # float --> (1,1,1)
        elif goal_form == "distance_to_goal":
            new_goal = torch.tensor(goal, device=device, dtype=torch.float32).reshape(1, 1, 1)

        # right append new observation and goal
        observations = torch.cat([observations, new_obs], dim=1)
        goals = torch.cat([goals, new_goal], dim=1)

        # right append new timestep
        timesteps = torch.cat(
            [timesteps,
            torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        if done:
            break

    # collect measures
    episode_length = env.get_current_step()
    success = env.is_success()
    spl = env.get_spl()
    #softspl = env.get_softspl()

    return episode_length, success, spl, real_act_seqs #, softspl

# evaluate rnn policy for one episode
def evaluate_one_episode_rnn(
        episode,
        env,
        model,
        goal_form,
        rnn_hidden_size,
        sample,
        max_ep_len,
        device,
        goal_dimension, 
        goal_coord_system,
        action_type
    ):

    # turn model into eval mode and move to desired device
    model.eval()
    model.to(device=device)

    # reset env
    obs = env.reset(episode=episode, plan_shortest_path=False)
    obs_array = extract_observation(obs, env.observation_space.spaces)
    if goal_form == "rel_goal":
        goal = np.array(obs["pointgoal"], dtype="float32")
    elif goal_form == "distance_to_goal":
        goal = env.get_current_distance()
    elif goal_form == "abs_goal":
        goal_position = np.array(env.goal_position, dtype="float32")
        goal = goal_position_to_abs_goal(goal_position,
            goal_dimension, goal_coord_system) # (2,) or (3,)

    # a0 is -1, shape (1)
    actions = torch.ones((1), device=device, dtype=torch.long) * (-1)

    # print("Scene id: %s"%(episode.scene_id))
    # print("Goal position: %s"%(env.goal_position)) # (3,)
    # print("Start position: %s"%(env.start_position)) # (3,)

    # change shape and convert to torch tensor
    # o: (C,H,W) --> (1,C,H,W)
    obs_array = np.expand_dims(obs_array, axis=0)
    observations = torch.from_numpy(obs_array).to(device=device, dtype=torch.float32)
    # g: (goal_dim,) --> (1,goal_dim)
    if goal_form == "rel_goal" or goal_form == "abs_goal":
        goal = np.expand_dims(goal, axis=0)
        goals = torch.from_numpy(goal).to(device=device, dtype=torch.float32)
    # float --> (1,1)
    elif goal_form == "distance_to_goal":
        goals = torch.tensor(goal, device=device, dtype=torch.float32).reshape(1, 1)

    # h0 is 0, shape [1, B, hidden_size]
    h = torch.zeros(1, 1, rnn_hidden_size, dtype=torch.float32, device=device) 

    real_act_seqs = []
    # run under policy for max_ep_len step or done
    # keep all history steps, but only use context length to predict current action
    for t in range(max_ep_len):
        # predict according to the sequence from (s0,a0,r0) up to now (context)
        actions, h = model.get_action(
            observations,
            actions,
            goals,
            h,
            sample=sample,
        )
        # actions: [B,1] --> [B]
        actions = torch.squeeze(actions, 1)

        # get action on cpu
        actions_cpu = actions.detach().cpu().item()

        real_act_seqs.append(actions_cpu)
        
        # step the env according to the action, get new observation and goal
        if action_type == "cartesian":
            obs, _, done, _ = env.step(actions_cpu)
        elif action_type == "polar":
            obs, _, done, _ = env.step_one_polar_action(actions_cpu)
        else:
            print("Error: undefined action type: %s"%(action_type))

        obs_array = extract_observation(obs, env.observation_space.spaces)
        if goal_form == "rel_goal":
            goal = np.array(obs["pointgoal"], dtype="float32")
        elif goal_form == "distance_to_goal":
            goal = env.get_current_distance()
        elif goal_form == "abs_goal":
            goal_position = np.array(env.goal_position, dtype="float32")
            goal = goal_position_to_abs_goal(goal_position,
            goal_dimension, goal_coord_system) # (2,) or (3,)

        # change shape and convert to torch tensor
        # (C,H,W) --> (1,C,H,W)
        obs_array = np.expand_dims(obs_array, axis=0)
        observations = torch.from_numpy(obs_array).to(device=device, dtype=torch.float32)
        # (goal_dim,) --> (1,goal_dim)
        if goal_form == "rel_goal" or goal_form == "abs_goal":
            goal = np.expand_dims(goal, axis=0)
            goals = torch.from_numpy(goal).to(device=device, dtype=torch.float32)
        # float --> (1,1)
        elif goal_form == "distance_to_goal":
            goals = torch.tensor(goal, device=device, dtype=torch.float32).reshape(1, 1)

        if done:
            break

    # collect measures
    episode_length = env.get_current_step()
    success = env.is_success()
    spl = env.get_spl()
    #softspl = env.get_softspl()

    return episode_length, success, spl, real_act_seqs #, softspl

# evaluate mlp policy or mlp q function (dqn, mlp_sqn, mlp_bc) for one episode
def evaluate_one_episode_mlp_policy_q_function(
        episode,
        env,
        model,
        goal_form,
        sample,
        max_ep_len,
        device,
        goal_dimension, 
        goal_coord_system,
        q_function,
        action_type
    ):

    # turn model into eval mode and move to desired device
    model.eval()
    model.to(device=device)

    # reset env
    obs = env.reset(episode=episode, plan_shortest_path=False)
    obs_array = extract_observation(obs, env.observation_space.spaces)
    if goal_form == "rel_goal":
        goal = np.array(obs["pointgoal"], dtype="float32")
    elif goal_form == "distance_to_goal":
        goal = env.get_current_distance()
    elif goal_form == "abs_goal":
        goal_position = np.array(env.goal_position, dtype="float32")
        goal = goal_position_to_abs_goal(goal_position,
            goal_dimension, goal_coord_system) # (2,) or (3,)

    # a0 is -1, shape (1)
    actions = torch.ones((1), device=device, dtype=torch.long) * (-1)

    # print("Scene id: %s"%(episode.scene_id))
    # print("Goal position: %s"%(env.goal_position)) # (3,)
    # print("Start position: %s"%(env.start_position)) # (3,)

    # change shape and convert to torch tensor
    # o: (C,H,W) --> (1,C,H,W)
    obs_array = np.expand_dims(obs_array, axis=0)
    observations = torch.from_numpy(obs_array).to(device=device, dtype=torch.float32)
    # g: (goal_dim,) --> (1,goal_dim)
    if goal_form == "rel_goal" or goal_form == "abs_goal":
        goal = np.expand_dims(goal, axis=0)
        goals = torch.from_numpy(goal).to(device=device, dtype=torch.float32)
    # float --> (1,1)
    elif goal_form == "distance_to_goal":
        goals = torch.tensor(goal, device=device, dtype=torch.float32).reshape(1, 1)

    real_act_seqs = []
    # run under policy for max_ep_len step or done
    # keep all history steps, but only use context length to predict current action
    for t in range(max_ep_len):
        # predict according to the sequence from (s0,a0,r0) up to now (context)
        # a = arg max q
        if q_function:  # q network
            actions = model.get_action(
                observations,
                goals
            )
        # sample a according to the policy distribution
        else:   # policy network 
            actions = model.get_action(
                observations,
                goals,
                sample=sample
            )
        # actions: [B,1] --> [B]
        actions = torch.squeeze(actions, 1)

        # get action on cpu
        actions_cpu = actions.detach().cpu().item()

        real_act_seqs.append(actions_cpu)
        
        # step the env according to the action, get new observation and goal
        if action_type == "cartesian":
            obs, _, done, _ = env.step(actions_cpu)
        elif action_type == "polar":
            obs, _, done, _ = env.step_one_polar_action(actions_cpu)
        else:
            print("Error: undefined action type: %s"%(action_type))

        obs_array = extract_observation(obs, env.observation_space.spaces)
        if goal_form == "rel_goal":
            goal = np.array(obs["pointgoal"], dtype="float32")
        elif goal_form == "distance_to_goal":
            goal = env.get_current_distance()
        elif goal_form == "abs_goal":
            goal_position = np.array(env.goal_position, dtype="float32")
            goal = goal_position_to_abs_goal(goal_position,
            goal_dimension, goal_coord_system) # (2,) or (3,)

        # change shape and convert to torch tensor
        # (C,H,W) --> (1,C,H,W)
        obs_array = np.expand_dims(obs_array, axis=0)
        observations = torch.from_numpy(obs_array).to(device=device, dtype=torch.float32)
        # (goal_dim,) --> (1,goal_dim)
        if goal_form == "rel_goal" or goal_form == "abs_goal":
            goal = np.expand_dims(goal, axis=0)
            goals = torch.from_numpy(goal).to(device=device, dtype=torch.float32)
        # float --> (1,1)
        elif goal_form == "distance_to_goal":
            goals = torch.tensor(goal, device=device, dtype=torch.float32).reshape(1, 1)

        if done:
            break

    # collect measures
    episode_length = env.get_current_step()
    success = env.is_success()
    spl = env.get_spl()
    #softspl = env.get_softspl()

    return episode_length, success, spl, real_act_seqs #, softspl

# evaluate rnn-ppo for one episode
def evaluate_one_episode_ppo(
        episode,
        env,
        model,
        obs_transforms,
        max_ep_len,
        device,
        cache,
        config
    ):
    # set model to eval mode
    model.eval()
    
    # reset env
    obs = env.reset(episode=episode, plan_shortest_path=False)
    # initialize model data structures
    recurrent_hidden_states, not_done_masks, prev_actions = init_ppo_inputs(model=model, config=config, 
        num_envs=1, device=device)
        
    # print("Scene id: %s"%(episode.scene_id))
    # print("Goal position: %s"%(env.goal_position))
    # print("Start position: %s"%(env.start_position))
    
    real_act_seqs = []
    # run under policy for max_ep_len step or done
    for t in range(max_ep_len):
        batch = get_ppo_batch(observations=[obs], 
            device=device, 
            cache=cache, 
            obs_transforms=obs_transforms)
        
        actions, recurrent_hidden_states = ppo_act(model=model, 
            batch=batch, 
            recurrent_hidden_states=recurrent_hidden_states, 
            prev_actions=prev_actions, not_done_masks=not_done_masks, 
            deterministic=False)
            
        with torch.no_grad():
            prev_actions.copy_(actions)
        
        action = actions[0][0].item()

        real_act_seqs.append(action)

        obs, _, done, _ = env.step(action)

        not_done_masks = torch.tensor(
            [[not done]],
            dtype=torch.bool,
            device=device,
        )

        if done:
            break   

    # collect measures
    episode_length = env.get_current_step()
    success = env.is_success()
    spl = env.get_spl()

    return episode_length, success, spl, real_act_seqs

# evaluate an agent across scene single env
class AcrossEnvEvaluatorSingle(AcrossEnvBaseEvaluator):
    
    def create_env(self, config_filename):
        self.env = MultiNavEnv(config_file=config_filename)
    
    
    def evaluate_over_one_dataset(self, episodes, model, sample, split_name, logs):
        episode_length_array = MeasureHistory("episode_length")
        success_array = MeasureHistory("success")
        spl_array = MeasureHistory("spl")
        #soft_spl_array = MeasureHistory("soft_spl")

        for i, episode in enumerate(episodes):
            #print('Episode: {}'.format(i+1))
            
            # only support cartesian action space
            if self.algorithm_name == "dt":
                episode_length, success, spl, real_act_seqs = evaluate_one_episode_dt(
                episode,
                self.env,
                model,
                self.goal_form,
                sample,
                self.max_ep_len,
                self.device,
                int(self.config.get("goal_dimension")), 
                self.config.get("goal_coord_system"))
            # support cartesian or polar action space
            elif self.algorithm_name == "rnn_bc":
                rnn_hidden_size = int(self.config.get("rnn_hidden_size"))
                episode_length, success, spl, real_act_seqs = evaluate_one_episode_rnn(
                episode,
                self.env,
                model,
                self.goal_form,
                rnn_hidden_size,
                sample,
                self.max_ep_len,
                self.device,
                int(self.config.get("goal_dimension")), 
                self.config.get("goal_coord_system"),
                action_type=self.action_type)
            # only support cartesian action space
            elif self.algorithm_name == "ppo":
                episode_length, success, spl, real_act_seqs = evaluate_one_episode_ppo(
                episode,
                self.env,
                model,
                self.obs_transforms,
                self.max_ep_len,
                self.device,
                self.cache,
                self.config)
            # support cartesian or polar action space
            elif self.algorithm_name == "mlp_bc":
                episode_length, success, spl, real_act_seqs = evaluate_one_episode_mlp_policy_q_function(
                episode,
                self.env,
                model,
                self.goal_form,
                sample,
                self.max_ep_len,
                self.device,
                int(self.config.get("goal_dimension")), 
                self.config.get("goal_coord_system"),
                q_function=False,
                action_type=self.action_type)
            # support cartesian or polar action space
            elif "dqn" in self.algorithm_name or "mlp_sqn" in self.algorithm_name:
                episode_length, success, spl, real_act_seqs = evaluate_one_episode_mlp_policy_q_function(
                episode,
                self.env,
                model,
                self.goal_form,
                sample,
                self.max_ep_len,
                self.device,
                int(self.config.get("goal_dimension")), 
                self.config.get("goal_coord_system"),
                q_function=True,
                action_type=self.action_type)
            else:
                print("Error: undefined algorithm name: %s"%(self.algorithm_name))
                exit()
            
            episode_length_array.add(episode_length)
            success_array.add(float(success))
            spl_array.add(spl)
            #soft_spl_array.add(softspl)

            # print episode info
            self.print_episode_info(episode_index=i, 
            env=self.env, 
            real_act_seqs=real_act_seqs, 
            episode_length=episode_length, 
            success=success, spl=spl)
        
        
        logs[f"{split_name}/total_episodes"] = success_array.len()
        logs[f"{split_name}/success_rate"] = success_array.mean()
        logs[f"{split_name}/spl"] = spl_array.mean()
        #logs[f"{split_name}/mean_soft_spl"] = soft_spl_array.mean()
        
        return logs
    
    def evaluate_over_datasets(self, checkpoint_file=None, model=None, sample=True):
        if model is None:
            model = self.load_model(checkpoint_file)
            
        
        logs = {}
        for split_name, episodes in self.eval_dataset_episodes.items():
            logs = self.evaluate_over_one_dataset(episodes, model, sample, split_name, logs)
        
        onecheckpoint_eval_results = {}
        return logs, onecheckpoint_eval_results

    def print_episode_info(self, episode_index, 
        env, real_act_seqs, episode_length, success, spl, compare_with_shortest_path=False):
        
        same_act_seqs = False
        if compare_with_shortest_path and env.optimal_action_seq and self.action_type == "cartesian":
            if success:
                if len(real_act_seqs) == len(env.optimal_action_seq):
                    same_flag = True
                    for i, real_act in enumerate(real_act_seqs):
                        if real_act != env.optimal_action_seq[i]:
                            same_flag = False
                            break
                    if same_flag:
                        same_act_seqs = True
        
        print("==================================")
        print("Episode: %d"%(episode_index+1))
        print("Success: %s"%(success))
        print("SPL: %f"%(spl))
        print("Episode length: %d"%(episode_length))
        if compare_with_shortest_path and env.optimal_action_seq and self.action_type == "cartesian":
            if success:
                print("Same with the optimal action sequence: %s"%(same_act_seqs))
                if not same_act_seqs:
                    print("Demonstration actions\n: %s"%env.optimal_action_seq)
                    print("Real actions\n: %s"%real_act_seqs)
        print("==================================")


# evaluate RNN BC or PPO in a single process here
if __name__ == "__main__":
    eval_splits = ["same_start_goal_val", "same_scene_val", "across_scene_val"]
    #eval_splits = ["same_scene_val", "across_scene_val"]
    #evaluator = AcrossEnvEvaluatorSingle(eval_splits=eval_splits, config_filename="imitation_learning_rnn_bc.yaml") 
    evaluator = AcrossEnvEvaluatorSingle(eval_splits=eval_splits, config_filename="pointgoal_ppo_multi_envs.yaml") 
    #evaluator.evaluate_over_checkpoints(sample=True)
    evaluator.plot_checkpoint_graphs()
    

        

    