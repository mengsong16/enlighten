#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import numpy as np
import torch

#from habitat_baselines.common.tensor_dict import TensorDict
from enlighten.agents.common.tensor_dict import TensorDict

# a buffer stores a segment of trajectory 
# for on-policy algorithms, this is equal to store one segment rollout collected by num_envs workers
# buffer length: numsteps
# support one or two buffer
# very specific to recurrent policy 
class RolloutStorage:
    r"""Class for storing rollout information for RL trainers."""

    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        is_double_buffered: bool = False,
    ):
        self.buffers = TensorDict()
        self.buffers["observations"] = TensorDict()

        for sensor in observation_space.spaces:
            self.buffers["observations"][sensor] = torch.from_numpy(
                np.zeros(
                    (
                        numsteps + 1,
                        num_envs,
                        *observation_space.spaces[sensor].shape,
                    ),
                    dtype=observation_space.spaces[sensor].dtype,
                )
            )

        self.buffers["recurrent_hidden_states"] = torch.zeros(
            numsteps + 1,
            num_envs,
            num_recurrent_layers,
            recurrent_hidden_state_size,
        )

        self.buffers["rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["value_preds"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["returns"] = torch.zeros(numsteps + 1, num_envs, 1)

        self.buffers["action_log_probs"] = torch.zeros(
            numsteps + 1, num_envs, 1
        )
        
        # if action_space.__class__.__name__ == "ActionSpace":
        #     action_shape = 1
        # else:
        #     action_shape = action_space.shape[0]

        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1    
        elif action_space.__class__.__name__ == "Discrete":
            action_shape = 1 # action is represented as index
        else:  # Box
            action_shape = action_space.shape[0]

        self.buffers["actions"] = torch.zeros(
            numsteps + 1, num_envs, action_shape
        )
        self.buffers["prev_actions"] = torch.zeros(
            numsteps + 1, num_envs, action_shape
        )

        # when action_space name is "ActionSpace", the action space is a dictionary of spaces
        if action_space.__class__.__name__ == "ActionSpace":
            self.buffers["actions"] = self.buffers["actions"].long()
            self.buffers["prev_actions"] = self.buffers["prev_actions"].long()

        # mask == True: end of episode
        self.buffers["masks"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        # one or two buffers
        # actually there is only one buffer structure, but could be used as two by pointers
        self.is_double_buffered = is_double_buffered
        self._nbuffers = 2 if is_double_buffered else 1

        self._num_envs = num_envs

        # each buffer should have the same number of env copies
        assert (self._num_envs % self._nbuffers) == 0

        self.numsteps = numsteps

        # every buffer has a current step index counter, initialize them to 0
        self.current_rollout_step_idxs = [0 for _ in range(self._nbuffers)]

    # each buffer's counter should be equal
    # return buffer 0's current step counter
    @property
    def current_rollout_step_idx(self) -> int:
        assert all(
            s == self.current_rollout_step_idxs[0]
            for s in self.current_rollout_step_idxs
        )
        return self.current_rollout_step_idxs[0]

    # cpu --> gpu or gpu --> cpu
    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))

    # insert one step data (a_t, r_t, v_t, s_{t+1}) to the specific buffer
    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
    ):
        if not self.is_double_buffered:
            assert buffer_index == 0

        next_step = dict(
            observations=next_observations,
            recurrent_hidden_states=next_recurrent_hidden_states,
            prev_actions=actions,
            masks=next_masks,
        )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
        )

        # filter out None values from the dictionaries
        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(
            int(buffer_index * self._num_envs / self._nbuffers),
            int((buffer_index + 1) * self._num_envs / self._nbuffers),
        )

        # insert data to current step and the next step loction of the buffer
        if len(next_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index] + 1, env_slice),
                next_step,
                strict=False,
            )

        if len(current_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index], env_slice),
                current_step,
                strict=False,
            )

    # advance the counter of the specific buffer
    def advance_rollout(self, buffer_index: int = 0):
        self.current_rollout_step_idxs[buffer_index] += 1

    # copy current step to step 0, clear the buffer after step 0, set current step counter to step 0
    # called when the policy is updated
    def after_update(self):
        self.buffers[0] = self.buffers[self.current_rollout_step_idx]

        self.current_rollout_step_idxs = [
            0 for _ in self.current_rollout_step_idxs
        ]

    # compute "returns" from step 0 to current step
    # returns: A(s,a)+V(s)
    # value_preds: V(s)
    # for all buffers
    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.buffers["value_preds"][
                self.current_rollout_step_idx
            ] = next_value
            gae = 0
            for step in reversed(range(self.current_rollout_step_idx)):
                delta = (
                    self.buffers["rewards"][step]
                    + gamma
                    * self.buffers["value_preds"][step + 1]
                    * self.buffers["masks"][step + 1]
                    - self.buffers["value_preds"][step]
                )
                gae = (
                    delta + gamma * tau * gae * self.buffers["masks"][step + 1]
                )
                self.buffers["returns"][step] = (
                    gae + self.buffers["value_preds"][step]
                )
        else:
            self.buffers["returns"][self.current_rollout_step_idx] = next_value
            for step in reversed(range(self.current_rollout_step_idx)):
                self.buffers["returns"][step] = (
                    gamma
                    * self.buffers["returns"][step + 1]
                    * self.buffers["masks"][step + 1]
                    + self.buffers["rewards"][step]
                )
    # Assign buffer to minibatches
    # Then assign input advantages to minibatches 
    # Only keep the hidden states of step 0 in minibatches
    # generator: can only be iterated once, yield=return
    def recurrent_generator(self, advantages, num_mini_batch) -> TensorDict:
        num_environments = advantages.size(1)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )

        # randomly shuffle the environment, then assign them equaly to each mini batch  
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            batch = self.buffers[0 : self.current_rollout_step_idx, inds]
            batch["advantages"] = advantages[
                0 : self.current_rollout_step_idx, inds
            ]
            #print("-------********-------")
            # [128, 3, 1, 512] 
            # number of rolling steps
            # number of environments per minibatch: e.g. 6/2
            # 1
            # size of hidden states
            #print(batch["recurrent_hidden_states"].size())
            #print("-------********-------")
            # Only keep the hidden states of step 0 in minibatches [1,3,1,512]
            # batch["recurrent_hidden_states"] = batch[
            #    "recurrent_hidden_states"
            # ][0:1]
            # [128, 3, 1, 512] --> [384, 1, 512]
            yield batch.map(lambda v: v.flatten(0, 1))
        # return a iterator over [2, 384,1,512] when num_mini_batch=2  

