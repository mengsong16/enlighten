#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

from enlighten.utils.path import config_path

class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    
    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    _is_distributed: bool
    _obs_batching_cache: ObservationBatchingCache
    envs: VectorEnv
    agent: PPO
    actor_critic: Policy

    def __init__(self, config_filename=None):
        # initialize parent class
        super().__init__(config_filename)

        # resume from checkpoint if "checkpoint_folder" indicate an existing file
        resume_state = load_resume_state(self.config)

        # recover config from saved checkpoint
        if resume_state is not None:
            self.config = resume_state["config"]

        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []

        self._static_encoder = False
        self._encoder = None
        self._obs_space = None

        # Distributed if the world size would be
        # greater than 1
        self._is_distributed = get_distrib_size()[2] > 1
        self._obs_batching_cache = ObservationBatchingCache()

    @property
    def obs_space(self):
        if self._obs_space is None and self.envs is not None:
            self._obs_space = self.envs.observation_spaces[0]

        return self._obs_space

    @obs_space.setter
    def obs_space(self, new_obs_space):
        self._obs_space = new_obs_space

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        r"""All reduce helper method that moves things to the correct
        device and only runs if distributed
        """
        if not self._is_distributed:
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)

    def _setup_actor_critic_agent(self) -> None:
        r"""Sets up actor critic and agent for PPO.
        """
        logger.add_filehandler(self.config.get("log_file"))

        if self.config.get("goal_format") == "pointgoal" and self.config.get("goal_coord_system") == "polar":
            polar_point_goal = True
        else:
            polar_point_goal = False
            
        # if transform exists, apply it to observation space    
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        self.obs_space = observation_space
        
        # print('-----------------------')
        # print(observation_space)
        # print(self.obs_space)
        # print('-----------------------')

        self._goal_obs_space = self.envs.get_goal_observation_space()

        # create actor critic
        if self.config.get("visual_encoder") == "CNN":
            self.actor_critic = CNNPolicy(observation_space=observation_space, 
                goal_observation_space=self._goal_obs_space, 
                polar_point_goal=polar_point_goal,
                action_space=self.envs.action_spaces[0],
                hidden_size=int(self.config.get("hidden_size")))
        else:
            # normalize with running mean and var if rgb images exist
            # assume that 
            self.actor_critic = ResNetPolicy(observation_space=observation_space, 
                goal_observation_space=self._goal_obs_space, 
                polar_point_goal=polar_point_goal,
                action_space=self.envs.action_spaces[0],
                hidden_size=int(self.config.get("hidden_size")),
                normalize_visual_inputs="color_sensor" in observation_space) 

        self.actor_critic.to(self.device)

        # load pretrained model
        if self.config.get("pretrained_visual_encoder") or self.config.get("pretrained_whole_model"):
            pretrained_state = torch.load(
                self.config.get("pretrained_model_path"), map_location="cpu"
            )

        # load pretrained actor critic
        if self.config.get("pretrained_whole_model"):
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        # load pretrained visual encoder    
        elif self.config.get("pretrained_visual_encoder"):
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )
        # freeze visual encoder if it is static
        if not self.config.get("train_encoder"):
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)


        #if self.config.RL.DDPPO.reset_critic:
        # reset the critic linear layer
        nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
        nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        # create agent
        self.agent = (DDPPO if self._is_distributed else PPO)(
            actor_critic=self.actor_critic,
            clip_param=float(self.config.get("clip_param")),
            ppo_epoch=int(self.config.get("ppo_epoch")),
            num_mini_batch=int(self.config.get("num_mini_batch")),
            value_loss_coef=float(self.config.get("value_loss_coef")),
            entropy_coef=float(self.config.get("entropy_coef")),
            lr=float(self.config.get("lr")),
            eps=float(self.config.get("eps")),
            max_grad_norm=float(self.config.get("max_grad_norm")),
            use_normalized_advantage=self.config.get("use_normalized_advantage"),
        )

    # create vector envs and scene dataset
    def _init_envs(self, config=None):
        if config is None:
            config = self.config

        if config.get("single_scene") == True:
            self.envs = construct_envs_based_on_singel_scene(
                config,
                workers_ignore_signals=is_slurm_batch_job(),
            )
        else:    
            self.envs = construct_envs_based_on_dataset(
                config,
                workers_ignore_signals=is_slurm_batch_job(),
            )


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
                self.config.get("distrib_backend")
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            # TO DO: need to check whether this may make difference for multi-processes
            # set torch and simulator gpu id according to local rank, not config file
            self.config["torch_gpu_id"] = local_rank
            self.config["simulator_gpu_id"] = local_rank
            # set seed according to rank and total num of environments
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config["seed"] += (
                torch.distributed.get_rank() * int(self.config.get("num_environments"))
            )
            #self.config.freeze()

            # random.seed(self.config.get("seed"))
            # np.random.seed(self.config.get("seed"))
            # torch.manual_seed(self.config.get("seed"))

            # set seed (except env seed)
            set_seed_except_env_seed(self.config["seed"])

            # initlalize how many rollouts are done    
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
        self._init_envs()

        # use gpu or not
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.get("torch_gpu_id"))
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # make checkpoint dir
        if rank0_only() and not os.path.isdir(self.config.get("checkpoint_folder")):
            os.makedirs(self.config.get("checkpoint_folder"))

        # setput actor critic of agent
        self._setup_actor_critic_agent()
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        #obs_space = self._obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            # TO DO: static visual features as observations
            # obs_space = spaces.Dict(
            #     {
            #         "visual_features": spaces.Box(
            #             low=np.finfo(np.float32).min,
            #             high=np.finfo(np.float32).max,
            #             shape=self._encoder.output_shape,
            #             dtype=np.float32,
            #         ),
            #         **obs_space.spaces,
            #     }
            # )

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
        # get initial observations and transform them
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        # static encoder visual features
        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        # add observation to rollout buffer
        self.rollouts.buffers["observations"][0] = batch

        # initialize episode reward, stats
        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=self.config.get("reward_window_size"))
        )

        # time counter
        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    # save checkpoint
    @rank0_only
    @profiling_utils.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.get("checkpoint_folder"), file_name)
        )

    # load checkpoint
    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)


    # extract scalars from info
    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    # extract scalars from infos
    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    # execute a policy and step the env, push the data to rollout buffer
    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        print("--------compute--------")

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
            )

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        actions = actions.to(device="cpu")
        self.pth_time += time.time() - t_sample_action

        profiling_utils.range_pop()  # compute actions

        t_step_env = time.time()

        # send the command of step env
        for index_env, act in zip(
            range(env_slice.start, env_slice.stop), actions.unbind(0)
        ):
            self.envs.async_step_at(index_env, act.item())

        self.env_time += time.time() - t_step_env

        # add actions to rollout buffer
        self.rollouts.insert(
            next_recurrent_hidden_states=recurrent_hidden_states,
            actions=actions,
            action_log_probs=actions_log_probs,
            value_preds=values,
            buffer_index=buffer_index,
        )

    #  step the env and collect obs
    def _collect_environment_result(self, buffer_index: int = 0):
        print("--------collect---------")
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

        print("--------------pt 1-----------------")
        outputs = [
            # step env
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]
        print(outputs)
        print("--------------pt 2-----------------")

        # unwrap the results
        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        self.env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rewards = torch.tensor(
            rewards_l,
            dtype=torch.float,
            device=self.current_episode_reward.device,
        )
        rewards = rewards.unsqueeze(1)

        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.current_episode_reward.device,
        )
        done_masks = torch.logical_not(not_done_masks)

        self.current_episode_reward[env_slice] += rewards
        current_ep_reward = self.current_episode_reward[env_slice]
        self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
        self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore
        
        # extract scalars from infos
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)
            if k not in self.running_episode_stats:
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )

            self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

        self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)

        # get static visual features
        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        # insert obs to rollout buffer
        self.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self.rollouts.advance_rollout(buffer_index)

        self.pth_time += time.time() - t_update_stats

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

        self.agent.train()

        value_loss, action_loss, dist_entropy = self.agent.update(
            self.rollouts
        )

        self.rollouts.after_update()
        self.pth_time += time.time() - t_update_model

        return (
            value_loss,
            action_loss,
            dist_entropy,
        )

    # update stats after step the env
    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        stats = self._all_reduce(stats)

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
            stats /= torch.distributed.get_world_size()

            losses = {
                k: stats[i].item() for i, k in enumerate(loss_name_ordering)
            }

        if self._is_distributed and rank0_only():
            self.num_rollouts_done_store.set("num_done", "0")

        self.num_steps_done += count_steps_delta

        return losses

    # update training log
    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }
        if len(metrics) > 0:
            writer.add_scalars("metrics", metrics, self.num_steps_done)

        writer.add_scalars(
            "losses",
            losses,
            self.num_steps_done,
        )

        # log stats
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

        # load state
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.agent.load_state_dict(resume_state["state_dict"])
            self.agent.optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(
                requeue_stats["window_episode_stats"]
            )

        # create tensorboard
        with (
            TensorboardWriter(
                self.config.get("tensorboard_dir"), flush_secs=self.flush_secs
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
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

                self.agent.eval()
                count_steps_delta = 0
                profiling_utils.range_push("rollouts loop")

                # act one step
                profiling_utils.range_push("_collect_rollout_step")
                for buffer_index in range(self._nbuffers):
                    self._compute_actions_and_step_envs(buffer_index)

                # execuate a rollout
                for step in range(int(self.config.get("num_steps"))):
                    is_last_step = (
                        self.should_end_early(step + 1)
                        or (step + 1) == int(self.config.get("num_steps"))
                    )

                    for buffer_index in range(self._nbuffers):
                        # step env
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
                            # act
                            self._compute_actions_and_step_envs(buffer_index)

                    if is_last_step:
                        break

                profiling_utils.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                (
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent()

                if self.config.get("use_linear_lr_decay"):
                    lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    dict(value_loss=value_loss, action_loss=action_loss),
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)

                # checkpoint model
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

    # evaluate checkpoint
    # TO DO
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        # use checkpoitn config or current config
        if self.config.get("eval_use_ckpt_config"):
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            #config = self.config.clone()
            config = copy.deepcopy(self.config)


        #config.defrost()
        #config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config["split"] = "val"
        #config.freeze()

        # TO DO
        # video needs top down map and collisions
        #if len(self.config.get("eval_video_option")) > 0:
            #config.defrost()
            #config["measurements"].append("top_down_map") 
            #config["measurements"].append("collisions") 
            #config.freeze()

        if config.get("verbose"):
            logger.info(f"env config: {config}")

        self._init_envs(config)
        self._setup_actor_critic_agent()

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )

        test_recurrent_hidden_states = torch.zeros(
            int(self.config.get("num_environments")),
            self.actor_critic.net.num_recurrent_layers,
            self.config.get("hidden_size"),
            device=self.device,
        )
        prev_actions = torch.zeros(
            int(self.config.get("num_environments")),
            1,
            device=self.device,
            dtype=torch.long,
        )
        not_done_masks = torch.zeros(
            int(self.config.get("num_environments")),
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(int(self.config.get("num_environments")))
        ]  # type: List[List[np.ndarray]]
        if len(self.config.get("eval_video_option")) > 0:
            os.makedirs(self.config.get("video_dir"), exist_ok=True)

        number_of_eval_episodes = self.config.get("test_episode_count")
        # evaluate on all episodes in the dataset
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)  # type: ignore

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {}
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.get("eval_video_option")) > 0:
                        generate_video(
                            video_option=self.config.get("eval_video_option"),
                            video_dir=self.config.get("video_dir"),
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.get("eval_video_option")) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    rgb_frames[i].append(frame)

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward",
            {"average reward": aggregated_stats["reward"]},
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()

if __name__ == "__main__":
   trainer = PPOTrainer(config_filename=os.path.join(config_path, "navigate_with_flashlight.yaml"))
   trainer.train()