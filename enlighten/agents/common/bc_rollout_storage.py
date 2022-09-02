#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import numpy as np
import torch

#from habitat_baselines.common.tensor_dict import TensorDict
from enlighten.agents.common.tensor_dict import TensorDict

# a buffer stores a whole trajectory for each of num_envs envs
# for behavior cloning
# buffer length: numsteps
class BCRolloutStorage:
    r"""Class for storing rollout information for BC trainers."""

    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space_channel,
        observation_space_height,
        observation_space_width,
        goal_space,
        action_space
    ):
        self.buffer = TensorDict()
        
        # (L+1)*N*obs_size
        observation_space_shape = (observation_space_channel, observation_space_height, observation_space_width)
        self.buffer["observations"] = torch.zeros(
            numsteps + 1,
            num_envs,
            *observation_space_shape)

        # (L+1)*N*goal_size
        self.buffer["goals"] = torch.zeros(
            numsteps + 1,
            num_envs,
            *goal_space.shape,
        )
        
        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1    
        elif action_space.__class__.__name__ == "Discrete":
            action_shape = 1 # action is represented as index
        else:  # Box
            action_shape = action_space.shape[0]

        self.buffer["actions"] = torch.zeros(
            numsteps + 1, num_envs, action_shape
        )
        self.buffer["prev_actions"] = torch.zeros(
            numsteps + 1, num_envs, action_shape
        )

        # when action_space name is "ActionSpace", the action space is a dictionary of spaces
        if action_space.__class__.__name__ == "ActionSpace":
            self.buffer["actions"] = self.buffer["actions"].long()
            self.buffer["prev_actions"] = self.buffer["prev_actions"].long()

        # mask == True: end of episode
        self.buffer["masks"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        self._num_envs = num_envs

        self.numsteps = numsteps

        # initialize step index counter to 0
        self.current_rollout_step_idx = 0

    
    # cpu --> gpu or gpu --> cpu
    def to(self, device):
        self.buffer.map_in_place(lambda v: v.to(device))

    # insert one step data (a_t, o_{t+1}, g_{t+1}, d_{t+1}) to the buffer
    def insert(
        self,
        next_observations=None,
        next_goals=None,
        actions=None,
        next_masks=None
    ):

        next_step = dict(
            observations=next_observations,
            goals=next_goals,
            prev_actions=actions,
            masks=next_masks,
        )

        current_step = dict(
            actions=actions
        )

        # filter out None values from the dictionaries (e.g. actions)
        # None will be set to the default values and be replaced later
        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(0, self._num_envs)
        

        # insert data to current location and the next location of the buffer
        if len(next_step) > 0:
            self.buffer.set(
                (self.current_rollout_step_idx + 1, env_slice),
                next_step,
                strict=False,
            )

        if len(current_step) > 0:
            self.buffer.set(
                (self.current_rollout_step_idx, env_slice),
                current_step,
                strict=False,
            )

    # advance the counter of the buffer
    def advance_rollout(self):
        self.current_rollout_step_idx += 1

    # copy current step to step 0, clear the buffer after step 0, set current step counter to step 0
    # called when the policy is updated
    def after_update(self):
        self.buffer[0] = self.buffer[self.current_rollout_step_idx]
        self.current_rollout_step_idx = 0

    # get a batch for training from the buffer
    def batch_generator(self) -> TensorDict:
        # randomly shuffle the environments
        inds = torch.randperm(self._num_envs)
        batch = self.buffer[0 : self.current_rollout_step_idx, inds]
            
        yield batch.map(lambda v: v.flatten(0, 1))
        
        # return a iterator over [2, 384,1,512] when num_mini_batch=2  

