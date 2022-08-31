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
from enlighten.agents.common.rollout_storage import RolloutStorage
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
from enlighten.agents.models.rnn_seq_model import RNNSequenceModel
from enlighten.agents.common.other import get_obs_channel_num

class BCOnlineTrainer(PPOTrainer):
    
    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    _is_distributed: bool
    _obs_batching_cache: ObservationBatchingCache
    envs: VectorEnv
    agent: RNNSequenceModel

    def __init__(self, config_filename=None):
        super().__init__()
        assert config_filename is not None, "needs config file to initialize trainer"
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)
        self._flush_secs = 30

        self.agent = None
        self.envs = None
    

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
        self.agent = RNNSequenceModel(
            obs_channel = get_obs_channel_num(self.config),
            obs_width = int(self.config.get("image_width")), 
            obs_height = int(self.config.get("image_height")),
            goal_dim=int(self.config.get("goal_dimension")),
            goal_form=self.config.get("goal_form"),
            act_num=int(self.config.get("action_number")),
            max_ep_len=int(self.config.get("max_ep_len")),  
            rnn_hidden_size=int(self.config.get('rnn_hidden_size')), 
            obs_embedding_size=int(self.config.get('obs_embedding_size')), #512
            goal_embedding_size=int(self.config.get('goal_embedding_size')), #32
            act_embedding_size=int(self.config.get('act_embedding_size')), #32
            rnn_type=self.config.get('rnn_type'),
            supervise_value=self.config.get('supervise_value'),
            domain_adaptation=self.config.get('domain_adaptation')
        )

        self.agent.to(self.device)


    # initialize training, reset envs
    def _init_train(self):
        # distributed or not
        if self.config.get("force_distributed"):
            self._is_distributed = True

        # is slurm or not
        if is_slurm_batch_job():
            add_signal_handlers()

        # set gpu and seed in distributed mode
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
                torch.distributed.get_rank() * int(self.config.get("num_environments"))
            )
            
            # set seed (except env seed)
            set_seed_except_env_seed(self.config["seed"])

            # initialize how many rollouts are done    
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        # verbose the entire config setting
        if rank0_only() and self.config.get("verbose"):
            logger.info(f"config: {self.config}")

        # profiling setting
        profiling_utils.configure(
            capture_start_step=-1,
            num_steps_to_capture=-1,
        )

        # create vector envs and dataset
        self._init_envs(split_name="train")

        # set device to gpu or not
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.get("gpu_id"))
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # make checkpoint dir
        if rank0_only() and not os.path.isdir(os.path.join(checkpoints_path, self.config.get("experiment_name"))):
            os.makedirs(os.path.join(checkpoints_path, self.config.get("experiment_name")))

        # setup agent
        self._setup_agent()
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        # create rollout buffer
        self._combined_goal_obs_space = self.envs.get_combined_goal_obs_space()
        
        # use single or double buffer
        self._nbuffers = 2 if self.config.get("use_double_buffered_sampler") else 1
        self.rollouts = RolloutStorage(
            self.config.get("num_steps"),
            self.envs.num_envs,
            self._combined_goal_obs_space,
            self.envs.action_spaces[0],
            self.config.get("hidden_size"),
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=self.config.get("use_double_buffered_sampler"),
        )
        self.rollouts.to(self.device)

        # reset envs
        observations = self.envs.reset()

        # get initial observations
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        
        # add initial observations to rollout buffer
        self.rollouts.buffers["observations"][0] = batch

        # initialize current episode reward, running_episode_stats to 0
        # Note that running_episode_stats accumulate along all history episodes (never reset to 0)
        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        # stats in info will be added later
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        # stats is calculated every n episodes
        # n <= self.window_episode_stats
        # this is a double ended queue, exceed the length will be popped
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=self.config.get("reward_window_size"))
        )

        # time counter
        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    # execute a policy, push the actions to rollout buffer
    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        #print("--------compute--------")

        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_sample_action = time.time()

        # sample actions
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idxs[buffer_index],
                env_slice,
            ]

            # get action
            # act only one step
            profiling_utils.range_push("compute actions")
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            ) # here deterministic = False

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        actions = actions.to(device="cpu")
        #print("========== actions =======")
        #print(actions)
        #print("==========================")
        self.pth_time += time.time() - t_sample_action

        profiling_utils.range_pop()  # compute actions

        t_step_env = time.time()

        # send the command of step env
        for index_env, act in zip(
            range(env_slice.start, env_slice.stop), actions.unbind(0)
        ):
            #print("------ actual action --------")
            #print(act.item())
            #print("------ actual action --------")
            #print(type(act.item()))
            self.envs.async_step_at(index_env, {"action": act.item()})
            #self.envs.async_step_at(index_env, act.item())

        self.env_time += time.time() - t_step_env

        # add actions, values, hidden states to rollout buffer
        self.rollouts.insert(
            next_recurrent_hidden_states=recurrent_hidden_states,
            actions=actions,
            action_log_probs=actions_log_probs,
            value_preds=values,
            buffer_index=buffer_index,
        )

    #  step the env and collect obs
    def _collect_environment_result(self, buffer_index: int = 0):
        #print("--------collect---------")
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_step_env = time.time()
        # print("-------------------------------")
        # print(env_slice.start)
        # print(env_slice.stop)
        # print("-------------------------------")

        #print("--------------pt 1-----------------")
        outputs = [
            # step all envs for one step
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]
        
        #print("--------------pt 2-----------------")

        # unwrap the results
        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        self.env_time += time.time() - t_step_env

        t_update_stats = time.time()

        # batch are only observations (do not include hidden states, etc)
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        
        # get rewards
        rewards = torch.tensor(
            rewards_l,
            dtype=torch.float,
            device=self.current_episode_reward.device,
        )
        rewards = rewards.unsqueeze(1)

        # get done (whether episode ends)
        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.current_episode_reward.device,
        )
        done_masks = torch.logical_not(not_done_masks)

        #print(type(infos))
        #print(infos)
        #print(type(rewards))
        #print(rewards)

        # accumulate rewards for current episode
        self.current_episode_reward[env_slice] += rewards
        current_ep_reward = self.current_episode_reward[env_slice]
        # add tensorboard stats here
        # add episode reward if episode ends, otherwise add 0
        self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
        # count: number of episodes in the window
        self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore
        
        # extract scalars from infos and add to running_episode_stats
        # under tensorboard "Metrics"
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)
            # create item and initialize to 0 if not exist
            if k not in self.running_episode_stats:
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )

            # add spl or success if episode ends, otherwise add 0 (spl and success is returned every step)
            # when episode ends, if episode fail, both spl and success is 0, otherwise they are positive
            self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

        # reset current episode cummulative reward to 0 if current episode is already done
        self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)

        # get static visual features
        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        # insert obs to rollout buffer
        # hidden states are zero (by default)
        self.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self.rollouts.advance_rollout(buffer_index)

        self.pth_time += time.time() - t_update_stats

        # return the number of steps collected by all envs
        # which will be added to the step counter of environment interation steps
        return env_slice.stop - env_slice.start

    @profiling_utils.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(self):
        
        self._compute_actions_and_step_envs()
        
        return self._collect_environment_result()

    # train/update policy
    @profiling_utils.RangeContext("_update_agent")
    def _update_agent(self):
        t_update_model = time.time()
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idx
            ]

            next_value = self.actor_critic.get_value(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        self.rollouts.compute_returns(
            next_value, self.config.get("use_gae"), self.config.get("gamma"), self.config.get("tau")
        )

        # switch agent model to training mode
        self.agent.train()

        # update agent parameters for ppo_epoch epoches
        value_loss, action_loss, dist_entropy, kl_divergence = self.agent.update(
            self.rollouts
        )

        self.rollouts.after_update()
        self.pth_time += time.time() - t_update_model

        return (
            value_loss,
            action_loss,
            dist_entropy,
            kl_divergence
        )

    # update stats after step the env
    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        # move to correct device
        stats = self._all_reduce(stats)

        # append accumlated stats at current timestep to window_episode_stats
        # stats: accumlated from step 0 to current step t
        # window means a moving window along the history steps
        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        if self._is_distributed:
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

        if self._is_distributed and rank0_only():
            self.num_rollouts_done_store.set("num_done", "0")

        self.num_steps_done += count_steps_delta

        return losses

    # update training log
    # log on tensorboard
    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        # compute delta between the first step and the last step inside the window
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        # add loss to tensorboard
        for k,v in losses.items():
            writer.add_scalar("Loss/"+str(k), v, self.num_steps_done)

        # Check to see if there are any metrics
        # that haven't been logged yet
        # add metrics to tensorboard and averaged over "count" episodes
        # deltas["count"] is the actual # of episodes inside the window, the maximum window size is reward_window_size
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }
        if len(metrics) > 0:
            for k,v in metrics.items():
                writer.add_scalar("Metrics/"+str(k), v, self.num_steps_done)    

        # add reward to tensorboard and averaged over "count" episodes
        writer.add_scalar(
            "Reward/reward",
            #{"reward": deltas["reward"] / deltas["count"]},
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # log stats
        # log_interval: log every # updates
        if self.num_updates_done % self.config.get("log_interval") == 0:
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

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )

    def should_end_early(self, rollout_step) -> bool:
        if not self._is_distributed:
            return False
        # This is where the preemption of workers happens.  If a
        # worker detects it will be a straggler, it preempts itself!
        return (
            rollout_step
            >= int(self.config.get("num_steps")) * self.SHORT_ROLLOUT_THRESHOLD
        ) and int(self.num_rollouts_done_store.get("num_done")) >= (
            float(self.config.get("sync_frac")) * torch.distributed.get_world_size()
        )

    # train process
    @profiling_utils.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        self._init_train()

        count_checkpoints = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )
        
        # create tensorboard
        tensorboard_folder = os.path.join(root_path, self.config.get("tensorboard_dir"), self.config.get("experiment_name"))
        if not os.path.exists(tensorboard_folder):
            os.makedirs(tensorboard_folder)
        
        with (
            TensorboardWriter(
                tensorboard_folder, flush_secs=self.flush_secs
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            # training loop
            # loop one time: self.num_updates_done + 1
            while not self.is_done():
                profiling_utils.on_start_step()
                profiling_utils.range_push("train update")

                if self.config.get("use_linear_clip_decay"):
                    self.agent.clip_param = self.config.get("clip_param") * (
                        1 - self.percent_done()
                    )

                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                    )

                    save_resume_state(
                        dict(
                            state_dict=self.agent.state_dict(),
                            optim_state=self.agent.optimizer.state_dict(),
                            lr_sched_state=lr_scheduler.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_utils.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return
                
                # switch agent to evaluation mode
                self.agent.eval()
                count_steps_delta = 0
                profiling_utils.range_push("rollouts loop")

                # act one step (for all envs)
                profiling_utils.range_push("_collect_rollout_step")
                for buffer_index in range(self._nbuffers):
                    self._compute_actions_and_step_envs(buffer_index)

                # all envs execuate a rollout (of length num_steps)
                for step in range(int(self.config.get("num_steps"))):
                    is_last_step = (
                        self.should_end_early(step + 1)
                        or (step + 1) == int(self.config.get("num_steps"))
                    )

                    for buffer_index in range(self._nbuffers):
                        # step env for one step
                        count_steps_delta += self._collect_environment_result(
                            buffer_index
                        )

                        if (buffer_index + 1) == self._nbuffers:
                            profiling_utils.range_pop()  # _collect_rollout_step

                        if not is_last_step:
                            if (buffer_index + 1) == self._nbuffers:
                                profiling_utils.range_push(
                                    "_collect_rollout_step"
                                )
                            # act one step
                            self._compute_actions_and_step_envs(buffer_index)

                    if is_last_step:
                        break

                profiling_utils.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                # train agent for one time
                (
                    value_loss,
                    action_loss,
                    dist_entropy,
                    kl_divergence
                ) = self._update_agent()

                if self.config.get("use_linear_lr_decay"):
                    lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1

                # show value_loass and action_loss in tensorboard
                # count_steps_delta: how many envs in this buffer, e.g. 6
                # steps: count_steps_delta * 1
                losses = self._coalesce_post_step(
                    dict(value_loss=value_loss, action_loss=action_loss, kl_divergence=kl_divergence, dist_entropy=dist_entropy),
                    count_steps_delta,
                )
                # update tensor board
                self._training_log(writer, losses, prev_time)

                # checkpoint model
                # count_checkpoints starts from 0
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

    # evaluate a checkpoint on one dataset
    def _eval_checkpoint_one_dataset(
        self,
        split_name,
        checkpoint_idx: int = 0,
        save_text_results=True
    ) -> None:
        
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # create vec envs
        self._init_envs(split_name=split_name)
        
        # create ppo model
        obs_transforms = get_active_obs_transforms(self.config)
        model = load_ppo_model(config=self.config, 
            observation_space=self.envs.observation_spaces[0],
            goal_observation_space=self.envs.get_goal_observation_space(),
            action_space=self.envs.action_spaces[0],
            device=self.device,
            obs_transforms=obs_transforms)
        
        # set model to eval mode
        model.eval()

        # init data structures
        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode, episode id is a combination of episode id and scene id

        rgb_frames_placeholder = [
            [] for _ in range(int(self.config.get("num_environments")))
        ]  # type: List[List[np.ndarray]]
        
        # get number of eval episodes
        number_of_eval_episodes = self.get_number_of_eval_episodes()
        # get tqdm bar
        pbar = tqdm.tqdm(total=number_of_eval_episodes)

        # reset envs and get initial observations
        observations = self.envs.reset()
        # get batch 0
        batch = get_ppo_batch(observations=observations, 
                device=self.device, 
                cache=self._obs_batching_cache, 
                obs_transforms=obs_transforms)
        
        # initialize model data structures
        recurrent_hidden_states, not_done_masks, prev_actions = init_ppo_inputs(model=model, config=self.config, 
            num_envs=self.envs.num_envs, device=self.device)
            
        # evaluate all episodes
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            # get current episodes
            current_episodes = self.envs.current_episodes()

            # act agent
            actions, recurrent_hidden_states = ppo_act(model=model, 
                    batch=batch, 
                    recurrent_hidden_states=recurrent_hidden_states, 
                    prev_actions=prev_actions, not_done_masks=not_done_masks, 
                    deterministic=False)
            
            with torch.no_grad():
                prev_actions.copy_(actions)

            # Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to an int
            step_data = [{"action": a.item()} for a in actions.to(device="cpu")]

            # step envs
            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]

            # get new batch
            batch = get_ppo_batch(observations=observations, 
                    device=self.device, 
                    cache=self._obs_batching_cache, 
                    obs_transforms=obs_transforms)

            # get new done masks on cpu
            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            # get new rewards
            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)

            # compute return of current episode as a vector
            current_episode_reward += rewards

            # get next episodes (to check envs to pause)
            next_episodes = self.envs.current_episodes()

            # get envs to pause if the episode assigned to the environment has ended
            # if an env has paused, it won't be stepped 
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended (done=True)
                # only record an episode's metric when it ends
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {}
                    # record return
                    episode_stats["return"] = current_episode_reward[i].item()
                    # extract and record other evaluation metrics from infos, succeed, spl
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    # reset return as 0
                    current_episode_reward[i] = 0
                    # record stats for current episode
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

            # pause envs 
            # all returned values should be updated according to non-paused envs
            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames_placeholder,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames_placeholder,
            )
            
        # evaluation has done, collect and record metrics
        self.record_eval_metrics(stats_episodes=stats_episodes,
                    checkpoint_idx = checkpoint_idx,
                    split_name = split_name,
                    save_text_results=save_text_results)

        # close envs
        self.envs.close()

if __name__ == "__main__":
   trainer = BCOnlineTrainer(config_filename=os.path.join(config_path, "imitation_learning_online_rnn.yaml"))
   trainer.train()
   #trainer.eval()