#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import os
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from gym import spaces
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
from torch import Tensor

from habitat import logger
from enlighten.envs import VectorEnv
from enlighten.utils.image_utils import observations_to_image
from enlighten.agents.trainer.base_trainer import BaseRLTrainer
from enlighten.agents.trainer.ppo_trainer import PPOTrainer
from enlighten.utils.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from enlighten.agents.common.bc_rollout_storage import BCRolloutStorage
from enlighten.utils.tensorboard_utils import TensorboardWriter
from habitat_sim.utils import profiling_utils

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

from enlighten.agents.algorithms.ppo import PPO
from enlighten.agents.algorithms.ddppo import DDPPO
from enlighten.agents.models import Policy
from enlighten.agents.common.tensor_related import (
    ObservationBatchingCache,
    batch_obs,
)
from enlighten.utils.video_utils import generate_video

from enlighten.envs.vec_env import construct_envs_based_on_dataset, construct_envs_based_on_singel_scene

from enlighten.agents.models import CNNPolicy, ResNetPolicy

from enlighten.agents.common.seed import set_seed, set_seed_except_env_seed

import copy

#from enlighten.utils.path import config_path
from enlighten.utils.path import *

from enlighten.agents.evaluation.ppo_eval import *
from enlighten.envs.nav_env import NavEnv
from enlighten.envs import MultiNavEnv
from enlighten.utils.video_utils import generate_video, images_to_video, create_video, remove_jpg, BGR_mode
from enlighten.tasks.measures import Measurements
from enlighten.agents.models.rnn_seq_model import DDBC
from enlighten.agents.common.other import get_obs_channel_num

class BCOnlineTrainer(PPOTrainer):
    
    _is_distributed: bool
    _obs_batching_cache: ObservationBatchingCache
    envs: VectorEnv
    agent: DDBC

    def __init__(self, config_filename=None):

        assert config_filename is not None, "needs config file to initialize trainer"
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)

        self._flush_secs = 30
        

        self.agent = None
        self.envs = None

        #self.num_envs = int(self.config.get("num_environments"))
    

        # Distributed if the world size would be
        # greater than 1
        self._is_distributed = get_distrib_size()[2] > 1
        self._obs_batching_cache = ObservationBatchingCache()

    
    def _setup_agent(self) -> None:
        r"""Sets up agent for BC
        """

        # set up log
        log_path = os.path.join(checkpoints_path, self.config.get("experiment_name"), "train.log")
        log_folder = os.path.dirname(log_path)
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        logger.add_filehandler(log_path)

        # create agent model
        self.agent = DDBC(
            obs_channel = get_obs_channel_num(self.config),
            obs_width = int(self.config.get("image_width")), 
            obs_height = int(self.config.get("image_height")),
            goal_dim=int(self.config.get("goal_dimension")),
            goal_form=self.config.get("goal_form"),
            act_num=int(self.config.get("action_number")), 
            rnn_hidden_size=int(self.config.get('rnn_hidden_size')), 
            obs_embedding_size=int(self.config.get('obs_embedding_size')), #512
            goal_embedding_size=int(self.config.get('goal_embedding_size')), #32
            act_embedding_size=int(self.config.get('act_embedding_size')), #32
            rnn_type=self.config.get('rnn_type'),
            supervise_value=self.config.get('supervise_value'),
            device=self.device,
            temperature=float(self.config.get('temperature', 1.0))
        )

    # [HEIGHT X WIDTH x CHANNEL] --> [CHANNEL x HEIGHT X WIDTH]
    # CHANNEL = {1,3,4}
    # return observation space shape
    # assume "color_sensor" is in the observation space
    def extract_observation_space_shape(self):
        observation_space = self.envs.observation_spaces[0]
        
        n_channel = 0
       
        if "color_sensor" in observation_space.spaces:
            n_height = observation_space.spaces["color_sensor"].shape[0]
            n_width = observation_space.spaces["color_sensor"].shape[1]
            
            n_channel += observation_space.spaces["color_sensor"].shape[2]

        if "depth_sensor" in observation_space.spaces:
            n_channel += observation_space.spaces["depth_sensor"].shape[2]

        # check if observation is valid
        if n_channel == 0:
            print("Error: channel of observation input is 0")
            exit()
        
        return n_channel, n_height, n_width

    # [N,C,H,W]
    # [N,3]
    def get_observation_goal_batch(self, observations):
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )

        goal_batch = batch['pointgoal']
        obs_batch = batch['color_sensor'].permute(0,3,1,2)
        if "depth_sensor" in batch.keys():
            depth_batch = batch['depth_sensor'].permute(0,3,1,2)
            obs_batch  = torch.cat((obs_batch, depth_batch), 1)

        return obs_batch, goal_batch

    # initialize training, reset envs
    def _init_train(self):
        # distributed or not
        if self.config.get("force_distributed"):
            self._is_distributed = True

        # is slurm or not
        if is_slurm_batch_job():
            add_signal_handlers()

        # set gpu and seed in distributed mode
        # otherwise use the gpu id and seed in the config
        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.get("distrib_backend"), int(self.config.get("default_port"))
            )
            if rank0_only():
                logger.info(
                    "Initialized BC with {} workers".format(
                        torch.distributed.get_world_size()  # world_size – Number of processes participating in the job
                    )
                )

            # set gpu id according to local rank, not config file
            self.config["gpu_id"] = local_rank
            # set seed according to rank and total num of environments
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config["seed"] += (
                torch.distributed.get_rank() * int(self.config.get("num_environments")) #self.num_envs
            )
            
            # set seed (except env seed)
            set_seed_except_env_seed(self.config["seed"])


        # verbose the entire config setting
        if rank0_only() and self.config.get("verbose"):
            logger.info(f"config: {self.config}")

        # profiling setting
        profiling_utils.configure(
            capture_start_step=-1,
            num_steps_to_capture=-1,
        )

        # create vector envs and dataset for training
        self._init_envs(split_name="train", auto_reset_done=False)

        # set device to gpu or not
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.get("gpu_id"))
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # make checkpoint dir
        if rank0_only() and not os.path.isdir(os.path.join(checkpoints_path, self.config.get("experiment_name"))):
            os.makedirs(os.path.join(checkpoints_path, self.config.get("experiment_name")))

        # setup agent (after setup device)
        self._setup_agent()
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        n_channel, n_height, n_width = self.extract_observation_space_shape()
        # create rollout buffer
        self.rollouts = BCRolloutStorage(
            numsteps=int(self.config.get("rollout_buffer_length")),
            num_envs=self.envs.num_envs, #int(self.config.get("num_environments")),
            observation_space_channel=n_channel,
            observation_space_height=n_height,
            observation_space_width=n_width,
            goal_space=self.envs.get_goal_observation_space()
        )
        self.rollouts.to(self.device)

        # reset envs
        observations = self.envs.reset()
        # each env plans its shortest path
        self.envs.plan_shortest_path()

        # get initial observations
        obs_batch, goal_batch = self.get_observation_goal_batch(observations)


        # add initial observations to rollout buffer
        self.rollouts.buffer["observations"][0] = obs_batch
        self.rollouts.buffer["goals"][0] = goal_batch
    

        # time counter
        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    # get whether all actions in a sequence have been executed
    def get_action_sequence_dones(self, cur_act_seq_lengths):
        cur_act_seq_lengths_list = list(cur_act_seq_lengths)

        dones = torch.tensor(
            [[seq_len <= 0] for seq_len in cur_act_seq_lengths_list],
            dtype=torch.bool,
            device=self.device,
        )

        return dones

    # once an env has been paused, it will be paused forever unless resumed
    @staticmethod
    def _pause_envs(
        dones,
        envs: Union[VectorEnv, NavEnv]
    ) -> Tuple[
        Union[VectorEnv, NavEnv]
    ]:
        # pausing self.envs which has done
        for idx, done in enumerate(dones):
            if done:
                envs.pause_at(idx)

        return envs

    # execute the optimal policy, save the actions to rollout buffer
    def _compute_actions_and_step_envs(self, cur_act_seq_lengths):
        #act_seq_dones = self.get_action_sequence_dones(cur_act_seq_lengths)
        #print(act_seq_dones)
        #self._pause_envs(act_seq_dones, self.envs)

        num_envs = self.envs.num_envs
        env_slice = slice(0, num_envs)

        # get optimal actions
        t_sample_action = time.time()
        
        # actions is a list (may contain None)
        actions = self.envs.get_next_optimal_action()

        #print(actions)
        
        self.pth_time += time.time() - t_sample_action

        profiling_utils.range_pop()  # compute actions

        # step the environments
        t_step_env = time.time()
        for index_env, act in zip(
            range(env_slice.start, env_slice.stop), actions
        ):
            # step the env
            self.envs.async_step_at(index_env, {"action": act})

        self.env_time += time.time() - t_step_env

        # action from list to tensor [B,1]
        actions = np.array(actions, dtype=int)
        actions = torch.from_numpy(actions).to(dtype=torch.long, device=self.device)
        actions = torch.unsqueeze(actions, dim=1)


        # add actions (a_t, a_{t+1}) to rollout buffer
        self.rollouts.insert(
            actions=actions
        )
        
    #  step the env and collect data
    def _collect_environment_result(self):
        #print("--------collect---------")
        num_envs = self.envs.num_envs
        # set env_slice
        env_slice = slice(0, num_envs)
        # start collection
        t_step_env = time.time()
        
        outputs = [
            # receive step results from all envs
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]
    
        # unwrap the results
        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        self.env_time += time.time() - t_step_env

        # organize data
        t_update_stats = time.time()

        # get batch from observations
        obs_batch, goal_batch = self.get_observation_goal_batch(observations)
        
        # get rewards
        # rewards = torch.tensor(
        #     rewards_l,
        #     dtype=torch.float,
        #     device=self.device,
        # )
        # rewards = rewards.unsqueeze(1)

        # get dones (whether episodes end)
        # not_done_masks = torch.tensor(
        #     [[not done] for done in dones],
        #     dtype=torch.bool,
        #     device=self.device,
        # )
        # done_masks = torch.logical_not(not_done_masks)

        # insert (o_t, g_t) to rollout buffer
        self.rollouts.insert(
            next_observations=obs_batch,
            next_goals=goal_batch
        )
        # rollout index++
        self.rollouts.advance_rollout()

        self.pth_time += time.time() - t_update_stats

        # return the number of steps collected by all envs
        # which will be added to the step counter of environment interation steps
        return env_slice.stop - env_slice.start

    # update agent for one time
    @profiling_utils.RangeContext("_update_agent")
    def _update_agent(self):
        t_update_model = time.time()
        
        # get training batch
        observations, action_targets, prev_actions, goals, batch_sizes = self.rollouts.get_training_batch(self.device)
        # print(action_targets)
        # print("==============================")
        # print(prev_actions)
        # print("==============================")
        # exit()

        # switch agent model to training mode
        self.agent.train()

        # forward agent
        rnn_hidden_size = int(self.config.get('rnn_hidden_size'))
        h_0 = torch.zeros(1, self.envs.num_envs, rnn_hidden_size, dtype=torch.float32, device=self.device) 
        action_preds = self.agent.actor.forward(observations, prev_actions, goals, h_0, batch_sizes)

        # update agent parameters for ppo_epoch epoches
        # action loss is computed over the whole sequence
        # action_preds: [T, action_num]
        # action_target are ground truth action indices (not one-hot vectors)
        action_loss =  F.cross_entropy(action_preds, action_targets)


        self.pth_time += time.time() - t_update_model

        return action_loss

    def clear_rollout_buffer_reset_envs(self):
        # clear rollouts
        self.rollouts.after_update()
        
        # reset envs
        self.envs.reset()
        # each env plans its shortest path
        self.envs.plan_shortest_path()

    # update stats after each update
    # count_steps_delta: the number of steps collected for this update
    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        
        # initialize stats
        stats = torch.zeros(1)
        stats = self._all_reduce(stats)

        if self._is_distributed:
            # average losses over all processes
            loss_name_ordering = sorted(losses.keys())
            stats = torch.tensor(
                [losses[k] for k in loss_name_ordering] + [count_steps_delta],
                device="cpu",
                dtype=torch.float32,
            )
            stats = self._all_reduce(stats)
            count_steps_delta = int(stats[-1].item())
            # get_world_size(): Returns the number of processes in the current process group
            stats /= torch.distributed.get_world_size()

            losses = {
                k: stats[i].item() for i, k in enumerate(loss_name_ordering)
            }

        self.num_steps_done += count_steps_delta

        return losses

    def should_checkpoint(self) -> bool:
        # do not save at when 0 update is done, save when total_updates are done
        needs_checkpoint = (
            self.num_updates_done % self.checkpoint_interval
        ) == 0

        return needs_checkpoint

    # log on tensorboard
    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        
        # add loss to tensorboard
        # use num_updates_done as x axis
        for k,v in losses.items():
            writer.add_scalar("Loss/"+str(k), v, self.num_updates_done)

        # log_interval: log every # updates, log at percentage 0
        if self.num_updates_done % int(self.config.get("log_interval")) == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    self.num_steps_done
                    / ((time.time() - self.t_start) + prev_time),
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

    # collect a batch of trajectories
    def collect_trajectory_batch(self):
        
        cur_act_seq_lengths = self.envs.get_optimal_action_sequence_lengths()
        max_num_steps = max(cur_act_seq_lengths) # denote as n
        cur_act_seq_lengths = np.array(cur_act_seq_lengths, dtype=int)
        # start lengths
        # print(cur_act_seq_lengths)
        # print("======================")
        # rollout length = optimal action sequence length + 1
        self.rollouts.seq_lengths = torch.from_numpy(copy.deepcopy(cur_act_seq_lengths+1))
        
        #print(self.rollouts.seq_lengths)

        profiling_utils.range_push("rollouts loop")

        # act one step (for all envs), add a1
        # step env for one step, buffer index++
        # must be paired with collect_environment_result
        profiling_utils.range_push("_collect_rollout_step")
        self._compute_actions_and_step_envs(cur_act_seq_lengths)
        cur_act_seq_lengths -= 1
        #print(cur_act_seq_lengths)
        #print("======================")

        count_steps_delta = 0
        # all envs execuate a rollout, iterate for n-1 times
        for step_ind in range(max_num_steps-1):
            # receive the last env step, add o_t, g_t (t=1,2,...n-1)
            count_steps_delta += self._collect_environment_result()

            profiling_utils.range_pop()  
            profiling_utils.range_push("_collect_rollout_step")

            # act one step, add a_{t+1} (t=2,3...,n)
            # step env for one step, buffer index++
            # must be paired with collect_environment_result
            self._compute_actions_and_step_envs(cur_act_seq_lengths)
            cur_act_seq_lengths -= 1
            #print(cur_act_seq_lengths)
            #print("======================")

        # receive the last env step, add o_n, g_n 
        count_steps_delta += self._collect_environment_result()

        # effective steps in the buffer: n+1 steps, from 0 to n
        assert self.rollouts.current_rollout_step_idx == max_num_steps, "Error: rollout buffer index should be equal to max action sequence length"

        # print("================")
        # print(self.rollouts.buffer["actions"][self.rollouts.current_rollout_step_idx-1])
        # print(self.rollouts.buffer["actions"][self.rollouts.current_rollout_step_idx]) # n+1 step: stop (0)
        # print("================")
        # print(self.rollouts.buffer["prev_actions"][0]) # 0 step: -1
        # print(self.rollouts.buffer["prev_actions"][self.rollouts.current_rollout_step_idx])
        # end lengths
        # print(cur_act_seq_lengths)
        # print("======================")
        
        profiling_utils.range_pop()  # rollouts loop

        # return the number of steps collected
        return count_steps_delta

    # train process
    @profiling_utils.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        self._init_train()
        
        # create optimizer and scheduler
        if self.config.get("optimizer") == "AdamW":
            optimizer = torch.optim.AdamW(
                self.agent.parameters(),
                lr=float(self.config.get('learning_rate')),
                weight_decay=float(self.config.get('weight_decay')),
            )
        elif self.config.get("optimizer") == "Adam":
            optimizer = torch.optim.Adam(
                self.agent.parameters(),
                lr=float(self.config.get('learning_rate'))
            )
        else:
            print("Error: unknown optimizer: %s"%(self.config.get("optimizer")))
            exit()
        
        print("======> created optimizer: %s"%(self.config.get("optimizer")))
        
        scheduler = None
        
        # create tensorboard folder
        tensorboard_folder = os.path.join(root_path, self.config.get("tensorboard_dir"), self.config.get("experiment_name"))
        if not os.path.exists(tensorboard_folder):
            os.makedirs(tensorboard_folder)
        
        # start training
        self.num_updates_done = 0 # the number of training updates done
        self.num_steps_done = 0   # the number of steps collected
        count_checkpoints = 0 # checkpoint index starts from 0
        prev_time = 0
        self.total_updates = int(self.config.get('total_updates')) # total training times
        self.checkpoint_interval = int(self.config.get('save_every_updates'))

        with (
            TensorboardWriter(
                tensorboard_folder, flush_secs=self.flush_secs
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            # training loop
            # loop one time: self.num_updates_done + 1
            while self.num_updates_done < self.total_updates:
                profiling_utils.on_start_step()
                profiling_utils.range_push("train update")

                # save resume state every n updates
                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        prev_time=(time.time() - self.t_start) + prev_time,
                    )

                    # note that the saved state has more information than the saved checkpoint
                    # checkpoint only include state_dict and config
                    if scheduler is None:
                        state = dict(
                            state_dict=self.agent.state_dict(),
                            optim_state=optimizer.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        )
                    else:
                        state = dict(
                            state_dict=self.agent.state_dict(),
                            optim_state=optimizer.state_dict(),
                            lr_sched_state=scheduler.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        )    
                    # config here is just a parameter to extract filename
                    save_resume_state(state, self.config)

                # exit function in the middle
                if EXIT.is_set():
                    profiling_utils.range_pop()  # train update
                    self.envs.close()
                    requeue_job()
                    return
                
                # collect a batch of trajectories
                count_steps_delta = self.collect_trajectory_batch()
                
                # update agent once
                action_loss = self._update_agent()

                # optimize for one step
                optimizer.zero_grad()
                action_loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                # updates counter++
                self.num_updates_done += 1

                # average losses over all processes
                # count_steps_delta: the number of steps collected
                losses = self._coalesce_post_step(
                    dict(action_loss=action_loss),
                    count_steps_delta,
                )
                # show losses in tensorboard
                self._training_log(writer, losses, prev_time)

                # clear rollout buffer and reset envs
                self.clear_rollout_buffer_reset_envs()

                # save checkpoint
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_utils.range_pop()  # train update

            self.envs.close()
    

if __name__ == "__main__":
   trainer = BCOnlineTrainer(config_filename=os.path.join(config_path, "imitation_learning_online_rnn_bc.yaml"))
   trainer.train()