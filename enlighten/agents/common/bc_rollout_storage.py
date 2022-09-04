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
        goal_space
    ):
        self.buffer = TensorDict()
        
        # [L+1,B,obs_size], default 0
        observation_space_shape = (observation_space_channel, observation_space_height, observation_space_width)
        self.buffer["observations"] = torch.zeros(
            numsteps + 1,
            num_envs,
            *observation_space_shape, dtype=torch.float32)

        # [L+1,B,goal_size], default 0
        self.buffer["goals"] = torch.zeros(
            numsteps + 1,
            num_envs,
            *goal_space.shape, dtype=torch.float32
        )
        
        # [L+1,B,1], default 0 (STOP)
        self.buffer["actions"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.int
        ) * (-1)
        # [L+1,B,1], default -1 (unknown)
        self.buffer["prev_actions"] = torch.ones(
            numsteps + 1, num_envs, 1, dtype=torch.int
        ) * (-1)

        # batch_sizes, default 0
        self.batch_sizes = torch.zeros(num_envs, dtype=torch.long)
       
        self._num_envs = num_envs

        # max length
        self.buffer_length = numsteps

        # rollout lengths
        self.seq_lengths = np.zeros(self._num_envs, dtype=int)

        # initialize step index counter to 0
        self.current_rollout_step_idx = 0

    
    # cpu --> gpu or gpu --> cpu
    def to(self, device):
        self.buffer.map_in_place(lambda v: v.to(device))

    # insert one step data (a_t, o_{t+1}, g_{t+1}) to the buffer
    def insert(
        self,
        next_observations=None,
        next_goals=None,
        actions=None
    ):

        next_step = dict(
            observations=next_observations,
            goals=next_goals,
            prev_actions=actions
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

    # clear the buffer
    # called when the policy is updated
    def after_update(self):
        self.current_rollout_step_idx = 0

    # get a batch for training from the buffer
    def get_training_batch(self) -> TensorDict:
        # randomly shuffle the environments
        inds = torch.randperm(self._num_envs)
        batch = self.buffer[0 : self.current_rollout_step_idx, inds]
            
        yield batch.map(lambda v: v.flatten(0, 1))
        
        # return a iterator over [2, 384,1,512] when num_mini_batch=2  

