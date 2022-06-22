#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import random
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

import numpy as np
import torch
import os
from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete

from habitat import logger

from enlighten.agents.models import CNNPolicy, ResNetPolicy
from enlighten.agents.common.tensor_related import batch_obs
from enlighten.agents.common.seed import set_seed
from enlighten.utils.config_utils import parse_config
from enlighten.utils.path import *
from enlighten.envs import NavEnv

from enlighten.utils.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)

import copy

from enlighten.utils.video_utils import generate_video, images_to_video

class Agent:
    r"""Abstract class for defining agents which act inside :ref:`core.env.Env`.

    This abstract class standardizes agents to allow seamless benchmarking.
    """

    def reset(self) -> None:
        r"""Called before starting a new episode in environment."""
        raise NotImplementedError

    def act(
        self, observations: "Observations"
    ) -> Union[int, str, Dict[str, Any]]:
        r"""Called to produce an action to perform in an environment.

        :param observations: observations coming in from environment to be
            used by agent to decide action.
        :return: action to be taken inside the environment and optional action
            arguments.
        """
        raise NotImplementedError

class PPOAgent(Agent):
    def __init__(self,  config_file, observation_space, 
        goal_observation_space, action_space,
        random_agent=False, use_vec_env=False) -> None:
        
        self.config = parse_config(config_file)
        self.device = (
            torch.device("cuda:{}".format(int(self.config.get("torch_gpu_id"))))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        #self.device = torch.device("cpu")

        # number of envs
        self.use_vec_env = use_vec_env
        if use_vec_env:
            self.num_envs = int(self.config.get("num_environments"))
        else:
            self.num_envs = 1    

        # load checkpoint
        checkpoint_path = os.path.join(root_path, self.config.get("eval_checkpoint_folder"), self.config.get("experiment_name"), self.config.get("eval_checkpoint_file"))
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint at: "+str(checkpoint_path))
        else:
            print("Error: path does not exist: %s"%(checkpoint_path))
            exit()    

        
        if not random_agent:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            # use checkpoint config
            if self.config.get("eval_use_ckpt_config"):
                self.config = copy.deepcopy(ckpt["config"])
                print(self.config)
                print("=====> Loaded config from checkpoint")
                

        # initialize model
        if self.config.get("goal_format") == "pointgoal" and self.config.get("goal_coord_system") == "polar":
            polar_point_goal = True
        else:
            polar_point_goal = False 
   
        
        # if transform exists, apply it to observation space
        obs_transforms = get_active_obs_transforms(self.config)
        if len(obs_transforms) > 0:
            observation_space = apply_obs_transforms_obs_space(
                observation_space, obs_transforms
            )
               
        if self.config.get("state_coord_system") == "polar":
            polar_state = True
        else:
            polar_state = False  
     

        if self.config.get("visual_encoder") == "CNN":
           
            self.actor_critic = CNNPolicy(observation_space=observation_space, 
                goal_observation_space=goal_observation_space, 
                polar_point_goal=polar_point_goal,
                action_space=action_space,
                rnn_type=self.config.get("rnn_type"),
                attention_type=str(self.config.get("attention_type")),
                goal_input_location=str(self.config.get("goal_input_location")),
                hidden_size=int(self.config.get("hidden_size")),
                blind_agent = self.config.get("blind_agent"),
                rnn_policy = self.config.get("rnn_policy"),
                state_only = self.config.get("state_only"),
                polar_state = polar_state,
                cos_augmented_goal = self.config.get("cos_augmented_goal"),
                cos_augmented_state = self.config.get("cos_augmented_state")
                )
        else:
            # normalize with running mean and var if rgb images exist
            # assume that
            
            self.actor_critic = ResNetPolicy(observation_space=observation_space, 
                goal_observation_space=goal_observation_space, 
                polar_point_goal=polar_point_goal,
                action_space=action_space,
                rnn_type=self.config.get("rnn_type"),
                attention_type=str(self.config.get("attention_type")),
                goal_input_location=str(self.config.get("goal_input_location")),
                hidden_size=int(self.config.get("hidden_size")),
                normalize_visual_inputs="color_sensor" in observation_space,
                attention=self.config.get("attention"),
                blind_agent = self.config.get("blind_agent"),
                rnn_policy = self.config.get("rnn_policy"),
                state_only = self.config.get("state_only"),
                polar_state = polar_state,
                cos_augmented_goal = self.config.get("cos_augmented_goal"),
                cos_augmented_state = self.config.get("cos_augmented_state")) 

        self.actor_critic.to(self.device)


        # load model
        if not random_agent:
            #  Filter out only actor_critic weights
            #  i.e. Policy: including attention, critic, net (recurrent encoder)
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )
            logger.info("===> Checkpoint loaded")
            #print(ckpt["state_dict"].keys())
            
        else:
            logger.error(
                "===> Model checkpoint wasn't loaded, evaluating " "a random model."
            )
         
        # set to eval mode
        self.actor_critic.eval()

        # create data structures
        self.recurrent_hidden_states: Optional[torch.Tensor] = None
        self.not_done_masks: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None


    def reset(self) -> None:
        # initialize data structures
        # T=N=1
        # h0 = 0
        self.recurrent_hidden_states = torch.zeros(
            self.num_envs, # num of envs
            self.actor_critic.net.num_recurrent_layers,
            self.config.get("hidden_size"),
            device=self.device,
        )
        self.not_done_masks = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.bool
        )
        self.prev_actions = torch.zeros(
            self.num_envs, 1, dtype=torch.long, device=self.device
        )

    # o --> a
    def act(self, observations, dones=None, cache=None) -> Dict[str, int]:
        if self.use_vec_env == False:
            observations = [observations]
          
        batch = batch_obs(observations, device=self.device, cache=cache)
        
        if dones is not None:
            self.not_done_masks = torch.tensor(
                    [[not done] for done in dones],
                    dtype=torch.bool,
                    device=self.device,
                )


        #print("before act")
        #print(self.recurrent_hidden_states.size())
        #print(self.prev_actions.size())
        #print(self.not_done_masks.size())

        # get h1,h2,h3,...
        with torch.no_grad():
            (
                _,
                actions, # [4,1]
                _,
                self.recurrent_hidden_states, # h_{t-1}
            ) = self.actor_critic.act(
                batch,
                self.recurrent_hidden_states, # h_{t} # [4,1, 512]
                self.prev_actions,  # [4,1]
                self.not_done_masks, # [4,1]
                deterministic=False,
            )


            
            self.prev_actions.copy_(actions) 

            #  Make masks not done till reset (end of episode) will be called
            #self.not_done_masks.fill_(True)


        #print("after act")
        #print(actions.size())
        if self.use_vec_env: # on gpu
            return actions, self.recurrent_hidden_states, self.prev_actions
        else:   # on cpu    
            #[1,224,224]
            #print(attention_image.shape)
            #return {"action": actions[0][0].item()}
            # actions: ([[index]])
            if self.config.get("attention"):
                attention_image = self.actor_critic.get_resized_attention_map(
                    batch, self.recurrent_hidden_states, self.prev_actions, self.not_done_masks)    
        
                return actions[0][0].item(), attention_image.cpu().detach().numpy()
            else:
                return actions[0][0].item()


def test():
    env =  NavEnv(config_file=os.path.join(config_path, "replica_nav_state.yaml"))
    agent = PPOAgent(config_file=os.path.join(config_path, "replica_nav_state.yaml"), 
    observation_space=env.observation_space, goal_observation_space=env.get_goal_observation_space(),
    action_space=env.action_space)
    
    step = 0
    obs = env.reset()
    done = None
    agent.reset()
    print('-----------------------------')
    print('Reset')
    print('-----------------------------')

    
    for i in range(50): 
        #action = env.action_space.sample()
        action = agent.act(observations=obs, dones=done)
        obs, reward, done, info = env.step(action)

        print("Step: %d, Action: %d"%(step, action))

        step += 1
        if done:
            break    

    print('Done.')
    
    # benchmark = habitat.Benchmark(config_paths=args.task_config)
    # metrics = benchmark.evaluate(agent)

    # for k, v in metrics.items():
    #     logger.info("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    test()
