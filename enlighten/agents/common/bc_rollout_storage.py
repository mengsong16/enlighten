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
        # has to be long instead of int
        self.buffer["actions"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.long
        )
        # [L+1,B,1], default -1 (unknown)
        # has to be long instead of int
        self.buffer["prev_actions"] = torch.ones(
            numsteps + 1, num_envs, 1, dtype=torch.long
        ) * (-1)


        self._num_envs = num_envs

        # max length
        self.buffer_length = numsteps

        # rollout lengths
        self.seq_lengths = torch.zeros(numsteps, dtype=torch.int)

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

    # Note that batch sizes should be of type long instead of int
    def get_batch_sizes(self, sorted_lengths):
        batch_sizes = torch.zeros(sorted_lengths[0], dtype=torch.long)
        for length in sorted_lengths:
            batch_sizes[:length] += 1
        
        return batch_sizes

    def concat_seqs_columnwise(self, sorted_block, batch_sizes, device):
        
        total_num_steps = torch.sum(batch_sizes).item()
        #print(total_num_steps)

        # (T,C,H,W)
        o_batch = torch.zeros(
            total_num_steps,
            *self.buffer["observations"].size()[2:], dtype=torch.float32, device=device
        )
        # (T,goal_dim)
        g_batch = torch.zeros(
            total_num_steps,
            *self.buffer["goals"].size()[2:], dtype=torch.float32, device=device
        )

        # (T), default 0 (STOP)
        a_batch = torch.zeros(total_num_steps, dtype=torch.long, device=device) 
        
        # (T), default -1 (unknown)
        prev_a_batch = torch.ones(total_num_steps, dtype=torch.long, device=device) * (-1)
        
        # print(o_batch.size())
        # print(g_batch.size())
        # print(a_batch.size())
        # print(prev_a_batch.size())
        offset = 0
        for column_index, batch_size in enumerate(batch_sizes):
            batch_size = batch_size.item() 
            o_batch[offset:offset+batch_size, :,:,:] = sorted_block["observations"][column_index,0:batch_size,:,:,:]
            g_batch[offset:offset+batch_size, :] = sorted_block["goals"][column_index,0:batch_size,:]
            a_batch[offset:offset+batch_size] = sorted_block["actions"][column_index,0:batch_size,:].view(-1)
            prev_a_batch[offset:offset+batch_size] = sorted_block["prev_actions"][column_index,0:batch_size,:].view(-1)

            offset += batch_size

        #print("offset: %d"%(offset))
        assert offset == total_num_steps, "Error: offset index %d should be equal to the total number of steps %d"%(offset, total_num_steps)
        
        # print("====================================")
        # print(a_batch)
        # print("====================================")
        # print(batch_sizes)
        # exit()
        return o_batch, g_batch, a_batch, prev_a_batch

    # get a batch for training from the buffer
    def get_training_batch(self, device) -> TensorDict:
        # Append STOP=0 to actions
        for i in range(self._num_envs):
            self.buffer["actions"][self.seq_lengths[i]-1, i] = 0
        # sort the rollouts in descending order based on lengths
        #print(self.seq_lengths)
        sorted_lengths, sorted_indices = torch.sort(self.seq_lengths, descending=True)
        # print(sorted_indices)
        # print(sorted_lengths)
        # print(self.current_rollout_step_idx)
        
        sorted_block = self.buffer[0 : self.current_rollout_step_idx+1, sorted_indices]
        # for i in range(self._num_envs):
        #     print(self.buffer["actions"][0 : self.current_rollout_step_idx+1, i].view(-1))
        #     print("-------------------------------")
        # print("==============================")
        # for i in range(self._num_envs):
        #     print(sorted_block["actions"][0 : sorted_lengths[i], i].view(-1))
        #     print("-------------------------------")
        # print("==============================")
        
        batch_sizes = self.get_batch_sizes(sorted_lengths)
        #print(sorted_lengths)
        #print(batch_sizes.size())
        #print(self.current_rollout_step_idx)
        assert self.current_rollout_step_idx+1 == batch_sizes.size()[0]
        #print(sorted_indices)
        
        o_batch, g_batch, a_batch, prev_a_batch = self.concat_seqs_columnwise(sorted_block, batch_sizes, device)
            
        return o_batch, a_batch, prev_a_batch, g_batch, batch_sizes 

