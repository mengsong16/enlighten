import numpy as np
import torch

from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.agents.common.seed import set_seed_except_env_seed
from enlighten.agents.common.other import get_device
from enlighten.envs.multi_nav_env import MultiNavEnv
from enlighten.agents.common.other import get_obs_channel_num
from enlighten.datasets.il_data_gen import load_behavior_dataset_meta, extract_observation


# evaluate an agent in across scene env
class MultiEnvEvaluator():
    def __init__(self, model, eval_split, config_filename="imitation_learning.yaml", 
        env: MultiNavEnv = None, device=None):

        assert config_filename is not None, "needs config file to initialize trainer"
        
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)

        # seed everything except env
        self.seed = int(self.config.get("seed"))
        set_seed_except_env_seed(self.seed)

        # create env if None
        if env is None:
            self.env = MultiNavEnv(config_file=config_filename) 
        else:
            self.env = env
        
        # the agent model to be evaluated
        self.model = model

        # device
        if device is None:
            self.device = get_device(self.config)
        else:
            self.device = device 

        # max episode length
        self.max_ep_len = int(self.config.get("max_ep_len"))     

        # load episodes of behavior dataset for evaluation
        self.eval_episodes = load_behavior_dataset_meta(yaml_name=config_filename, 
            split_name=eval_split)

    def evaluate_over_dataset(self):
        for i, episode in enumerate(self.eval_episodes):
            print('Episode: {}'.format(i+1))
            episode_length, success, spl, softspl = self.evaluate_one_episode_dt(
                episode,
                state_dim,
                act_dim,
                model,
                sample,
                max_ep_len,
                device)

    # evaluate decision transformer for one episode
    def evaluate_one_episode_dt(self,
            episode,
            state_dim,
            act_dim,
            model,
            sample,
            max_ep_len,
            device
        ):

        # turn model into eval mode
        model.eval()
        model.to(device=device)

        # reset env
        obs = self.env.reset(episode=episode, plan_shortest_path=False)
        obs_array = extract_observation(obs, self.env.observation_space.spaces)
        rel_goal = np.array(obs["pointgoal"], dtype="float32")

        print("Scene id: %s"%(episode.scene_id))
        print("Goal position: %s"%(self.env.goal_position))
        print("Start position: %s"%(self.env.start_position))

        # note that the latest action and reward will be "padding"
        observations = torch.from_numpy(obs_array).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        # place-holder
        actions = torch.zeros((1, act_dim), device=device, dtype=torch.float32)
        goals = torch.zeros(0, device=device, dtype=torch.float32)

        ep_return = target_return
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        # run under policy for max_ep_len step
        for t in range(max_ep_len):

            # post pad a 0 to action sequence and reward sequence
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)

            # predict according to the sequence from (s0,a0,r0) up to now (context)
            # need to input timesteps as positional embedding
            action = model.get_action(
                observations.to(dtype=torch.float32),
                actions.to(dtype=torch.float32),
                goals.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                sample=sample,
            )
            
            # append new action
            actions[-1] = action
            
            # step the env according to action, get new observation and goal
            action = action.detach().cpu().numpy()
            obs, _, done, _ = self.env.step(action)
            obs_array = extract_observation(obs, self.env.observation_space.spaces)
            rel_goal = np.array(obs["pointgoal"], dtype="float32")

            # change shape and convert to torch tensor
            # (C,H,W) --> (1,C,H,W)
            obs_array = np.expand_dims(obs_array, axis=0)
            obs = torch.from_numpy(obs_array).to(device=device, dtype=torch.float32)
            
            # append new observation and goal
            observations = torch.cat([observations, obs], dim=0)
            goals[-1] = rel_goal

            # append target return
            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
        
            # append new timestep
            timesteps = torch.cat(
                [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)


            if done:
                break

        # collect measurs
        episode_length = self.env.get_current_step()
        success = self.env.is_success()
        spl = self.env.get_spl()
        softspl = self.env.get_softspl()

        return episode_length, success, spl, softspl

