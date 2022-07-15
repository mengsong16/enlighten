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
from enlighten.agents.evaluation.ppo_eval import *

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
        device
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
    # a0 is 0, shape (1,1)
    actions = torch.zeros((1, 1), device=device, dtype=torch.long)

    print("Scene id: %s"%(episode.scene_id))
    print("Goal position: %s"%(env.goal_position))
    print("Start position: %s"%(env.start_position))

    # change shape and convert to torch tensor
    # (C,H,W) --> (1,1,C,H,W)
    obs_array = np.expand_dims(np.expand_dims(obs_array, axis=0), axis=0)
    observations = torch.from_numpy(obs_array).to(device=device, dtype=torch.float32)
    # (goal_dim,) --> (1,1,goal_dim)
    if goal_form == "rel_goal":
        goal = np.expand_dims(np.expand_dims(goal, axis=0), axis=0)
        goals = torch.from_numpy(goal).to(device=device, dtype=torch.float32)
    # float --> (1,1,1)
    elif goal_form == "distance_to_goal":
        goals = torch.tensor(goal, device=device, dtype=torch.float32).reshape(1, 1, 1)

    # t0 is 0, shape (1,1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

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
        new_action = torch.tensor(action, device=device, dtype=torch.long).reshape(1, 1)
        actions = torch.cat([actions, new_action], dim=1)

        # step the env according to the action, get new observation and goal
        obs, _, done, _ = env.step(action)
        obs_array = extract_observation(obs, env.observation_space.spaces)
        if goal_form == "rel_goal":
            goal = np.array(obs["pointgoal"], dtype="float32")
        elif goal_form == "distance_to_goal":
            goal = env.get_current_distance()

        # change shape and convert to torch tensor
        # (C,H,W) --> (1,1,C,H,W)
        obs_array = np.expand_dims(np.expand_dims(obs_array, axis=0), axis=0)
        new_obs = torch.from_numpy(obs_array).to(device=device, dtype=torch.float32)
        # (goal_dim,) --> (1,1,goal_dim)
        if goal_form == "rel_goal":
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

    return episode_length, success, spl #, softspl

# evaluate rnn for one episode
def evaluate_one_episode_rnn(
        episode,
        env,
        model,
        goal_form,
        rnn_hidden_size,
        sample,
        max_ep_len,
        device
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
    # a0 is -1, shape (1)
    actions = torch.ones((1), device=device, dtype=torch.long) * (-1)

    print("Scene id: %s"%(episode.scene_id))
    print("Goal position: %s"%(env.goal_position))
    print("Start position: %s"%(env.start_position))

    # change shape and convert to torch tensor
    # o: (C,H,W) --> (1,C,H,W)
    obs_array = np.expand_dims(obs_array, axis=0)
    observations = torch.from_numpy(obs_array).to(device=device, dtype=torch.float32)
    # g: (goal_dim,) --> (1,goal_dim)
    if goal_form == "rel_goal":
        goal = np.expand_dims(goal, axis=0)
        goals = torch.from_numpy(goal).to(device=device, dtype=torch.float32)
    # float --> (1,1)
    elif goal_form == "distance_to_goal":
        goals = torch.tensor(goal, device=device, dtype=torch.float32).reshape(1, 1)

    # h0 is 0, shape [1, B, hidden_size]
    h = torch.zeros(1, 1, rnn_hidden_size, dtype=torch.float32, device=device) 

    # run under policy for max_ep_len step or done
    # keep all history steps, but only use context length to predict current action
    for t in range(max_ep_len):
        # predict according to the sequence from (s0,a0,r0) up to now (context)
        # need to input timesteps as positional embedding
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
        
        # step the env according to the action, get new observation and goal
        obs, _, done, _ = env.step(actions_cpu)
        obs_array = extract_observation(obs, env.observation_space.spaces)
        if goal_form == "rel_goal":
            goal = np.array(obs["pointgoal"], dtype="float32")
        elif goal_form == "distance_to_goal":
            goal = env.get_current_distance()

        # change shape and convert to torch tensor
        # (C,H,W) --> (1,C,H,W)
        obs_array = np.expand_dims(obs_array, axis=0)
        observations = torch.from_numpy(obs_array).to(device=device, dtype=torch.float32)
        # (goal_dim,) --> (1,goal_dim)
        if goal_form == "rel_goal":
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

    return episode_length, success, spl #, softspl

# evaluate ppo for one episode
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
        
    print("Scene id: %s"%(episode.scene_id))
    print("Goal position: %s"%(env.goal_position))
    print("Start position: %s"%(env.start_position))
    
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

    return episode_length, success, spl

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
            print('Episode: {}'.format(i+1))
            
            if self.algorithm_name == "dt":
                episode_length, success, spl = evaluate_one_episode_dt(
                    episode,
                    self.env,
                    model,
                    self.goal_form,
                    sample,
                    self.max_ep_len,
                    self.device)
            elif self.algorithm_name == "rnn":
                rnn_hidden_size = int(self.config.get("rnn_hidden_size"))
                episode_length, success, spl = evaluate_one_episode_rnn(
                episode,
                self.env,
                model,
                self.goal_form,
                rnn_hidden_size,
                sample,
                self.max_ep_len,
                self.device)
            elif self.algorithm_name == "ppo":
                episode_length, success, spl = evaluate_one_episode_ppo(
                    episode,
                    self.env,
                    model,
                    self.obs_transforms,
                    self.max_ep_len,
                    self.device,
                    self.cache,
                    self.config)
            else:
                print("Error: undefined algorithm name: %s"%(self.algorithm_name))
                exit()
            
            episode_length_array.add(episode_length)
            success_array.add(float(success))
            spl_array.add(spl)
            #soft_spl_array.add(softspl)
        
        
        logs[f"{split_name}/total_episodes"] = success_array.len()
        logs[f"{split_name}/success_rate"] = success_array.mean()
        logs[f"{split_name}/mean_spl"] = spl_array.mean()
        #logs[f"{split_name}/mean_soft_spl"] = soft_spl_array.mean()
        
        return logs
    
    def evaluate_over_datasets(self, model=None, sample=True):
        if model is None:
            model = self.load_model()
            
        
        logs = {}
        for split_name, episodes in self.eval_dataset_episodes.items():
            logs = self.evaluate_over_one_dataset(episodes, model, sample, split_name, logs)
        
        return logs

    

if __name__ == "__main__":
    eval_splits = ["same_start_goal_test", "same_scene_test", "across_scene_test"]
    #evaluator = AcrossEnvEvaluatorSingle(eval_splits=eval_splits, config_filename="imitation_learning_rnn.yaml") 
    evaluator = AcrossEnvEvaluatorSingle(eval_splits=eval_splits, config_filename="pointgoal_multi_envs.yaml") 
    logs = evaluator.evaluate_over_datasets(sample=True)
    evaluator.print_metrics(logs, eval_splits)
    evaluator.save_eval_logs(logs, eval_splits)

        

    