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
from enlighten.utils.utils import parse_config
from enlighten.utils.path import *
from enlighten.envs import NavEnv

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
    def __init__(self,  env, config_file=os.path.join(config_path, "navigate_with_flashlight.yaml"), load_model_path=None) -> None:
        self.hidden_size = 512
        self.env = env
        self.config = parse_config(config_file)
        self.device = (
            torch.device("cuda:{}".format(int(self.config.get("gpu_id"))))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        #self.device = torch.device("cpu")

        set_seed(seed=int(self.config.get("seed")), env=self.env)
        
        if self.config.get("goal_format") == "pointgoal" and self.config.get("goal_coord_system") == "polar":
            polar_point_goal = True
        else:
            polar_point_goal = False    

        if self.config.get("visual_encoder") == "CNN":
            self.actor_critic = CNNPolicy(observation_space=env.observation_space, 
                goal_observation_space=env.get_goal_observation_space(), 
                polar_point_goal=polar_point_goal,
                action_space=env.action_space,
                hidden_size=self.hidden_size)
        else:
            # normalize with running mean and var if rgb images exist
            # assume that 
            self.actor_critic = ResNetPolicy(observation_space=env.observation_space, 
                goal_observation_space=env.get_goal_observation_space(), 
                polar_point_goal=polar_point_goal,
                action_space=env.action_space,
                hidden_size=self.hidden_size,
                normalize_visual_inputs="color_sensor" in env.observation_space) 

        self.actor_critic.to(self.device)

        # load model
        if load_model_path:
            ckpt = torch.load(load_model_path, map_location=self.device)
            #  Filter only actor_critic weights
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )
        else:
            logger.error(
                "Model checkpoint wasn't loaded, evaluating " "a random model."
            )

        # create data structures
        self.recurrent_hidden_states: Optional[torch.Tensor] = None
        self.not_done_masks: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None

    def reset(self) -> None:
        # initialize data structures
        self.recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.net.num_recurrent_layers,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(
            1, 1, device=self.device, dtype=torch.bool
        )
        self.prev_actions = torch.zeros(
            1, 1, dtype=torch.long, device=self.device
        )

    # o --> a
    def act(self, observations) -> Dict[str, int]:
        batch = batch_obs([observations], device=self.device)
        with torch.no_grad():
            (
                _,
                actions,
                _,
                self.recurrent_hidden_states,
            ) = self.actor_critic.act(
                batch,
                self.recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(actions) 

        #return {"action": actions[0][0].item()}
        # actions: tensor([[index]])
        return actions[0][0].item()


def test():
    env =  NavEnv()
    agent = PPOAgent(env=env)
    #print(env.get_goal_observation_space())
    step = 0
    obs = env.reset()
    agent.reset()
    print('-----------------------------')
    print('Reset')
    print('-----------------------------')

    
    for i in range(50): 
        #action = env.action_space.sample()
        action = agent.act(obs)
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