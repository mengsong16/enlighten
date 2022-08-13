#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from typing import Any, ClassVar, Dict, List, Tuple, Union

import torch
from numpy import ndarray
from torch import Tensor

from habitat import logger

from enlighten.envs import VectorEnv, NavEnv
from enlighten.utils.tensorboard_utils import TensorboardWriter
from enlighten.utils.ddp_utils import SAVE_STATE, is_slurm_batch_job
from enlighten.agents.common.checkpoint import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

from enlighten.utils.config_utils import parse_config
from enlighten.utils.path import *

import copy

class BaseTrainer:
    r"""Generic trainer class that serves as a base template for more
    specific trainer classes like RL trainer, SLAM or imitation learner.
    Includes only the most basic functionality.
    """

    supported_tasks: ClassVar[List[str]]

    def train(self) -> None:
        raise NotImplementedError

    def merge_config1_to_config2(self, config1, config2):
        assert isinstance(config1, dict) and isinstance(config2, dict)
        for k, v in config1.items():
            config2[k] = v

        return config2    

    # TO DO: yaml config merge needs to be checked
    def _setup_eval_config(self, checkpoint_config):
        r"""Sets up and returns a merged config for evaluation. Config
            object saved from checkpoint is merged into config file specified
            at evaluation time with the following overwrite priority:
                eval_opts > ckpt_opts > eval_cfg > ckpt_cfg
            If the saved config is outdated, only the eval config is returned.

        Args:
            checkpoint_config: saved config from checkpoint.

        Returns:
            Config: merged config for eval.
        """

        #config = self.config.clone()
        config = copy.deepcopy(self.config)

        self.merge_config1_to_config2(checkpoint_config, config)

        # CMD_TRAILING_OPTS: store command line options as list of strings
        # ckpt_cmd_opts = checkpoint_config.CMD_TRAILING_OPTS
        # eval_cmd_opts = config.CMD_TRAILING_OPTS

        # try:
        #     config.merge_from_other_cfg(checkpoint_config)
        #     config.merge_from_other_cfg(self.config)
        #     config.merge_from_list(ckpt_cmd_opts)
        #     config.merge_from_list(eval_cmd_opts)
        # except KeyError:
        #     logger.info("Saved config is outdated, using solely eval config")
        #     config = self.config.clone()
        #     config.merge_from_list(eval_cmd_opts)
        #config.defrost()
        if config["split"] == "train":
            config["split"] = "val"
        #config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = self.config.SENSORS
        #config.freeze()

        return config

    def eval(self) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        or BaseILTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", int(self.config.get("torch_gpu_id")))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # video options when evaluate
        if "tensorboard" in self.config.get("eval_video_option"):
            assert (
                len(self.config.get("tensorboard_dir")) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(os.path.join(root_path, self.config.get("tensorboard_dir"), self.config.get("experiment_name")), exist_ok=True)
        if "disk" in self.config.get("eval_video_option"):
            if not os.path.exists(os.path.join(root_path, self.config.get("eval_dir"), self.config.get("experiment_name"))):
                os.makedirs(os.path.join(root_path, self.config.get("eval_dir"), self.config.get("experiment_name")), exist_ok=True) # exist_ok=True, won't raise any error if folder exists

        with TensorboardWriter(
            os.path.join(root_path, self.config.get("tensorboard_dir"), self.config.get("experiment_name")), flush_secs=self.flush_secs
        ) as writer:
            checkpoint_list = list(self.config.get("eval_checkpoint_file"))
            # evaluate checkpoints in the list provided in config
            if "*" not in checkpoint_list:
                for checkpoint_filename in checkpoint_list:
                    # evaluate a single checkpoint
                    single_checkpoint = os.path.join(root_path, self.config.get("eval_checkpoint_folder"), self.config.get("experiment_name"), checkpoint_filename)
                    proposed_index = get_checkpoint_id(single_checkpoint)
                    if proposed_index is not None:
                        ckpt_idx = proposed_index
                    else:
                        ckpt_idx = 0
                    self._eval_checkpoint(
                        single_checkpoint,
                        writer,
                        checkpoint_index=ckpt_idx,
                    )
            else:
                # evaluate all checkpoints in the checkpoint directory in order
                prev_ckpt_ind = -1
                while True:
                    current_ckpt = None
                    while current_ckpt is None:
                        # pull out current ckpt (prev_ckpt_ind+1)
                        current_ckpt = poll_checkpoint_folder(
                            os.path.join(root_path, self.config.get("eval_checkpoint_folder"), self.config.get("experiment_name")), prev_ckpt_ind
                        )
                        time.sleep(2)  # sleep for 2 secs before polling again
                    logger.info(f"=======current_ckpt: {current_ckpt}=======")
                    prev_ckpt_ind += 1
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind,
                    )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError


class BaseRLTrainer(BaseTrainer):
    r"""Base trainer class for RL trainers. Future RL-specific
    methods should be hosted here.
    """
    device: torch.device  # type: ignore
    video_option: List[str]
    num_updates_done: int
    num_steps_done: int
    _flush_secs: int
    _last_checkpoint_percent: float

    def __init__(self, config_filename) -> None:
        super().__init__()
        assert config_filename is not None, "needs config file to initialize trainer"
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)
        self._flush_secs = 30
        self.num_updates_done = 0
        self.num_steps_done = 0
        self._last_checkpoint_percent = -1.0

        self.num_updates = int(self.config.get("num_updates"))
        self.total_num_steps = int(self.config.get("total_num_steps"))
        self.num_checkpoints = int(self.config.get("num_checkpoints"))
        self.checkpoint_interval = int(self.config.get("checkpoint_interval"))
        self.validate_config_para()

        
    def validate_config_para(self):
        if self.num_updates != -1 and self.total_num_steps != -1:
            raise RuntimeError(
                "NUM_UPDATES and TOTAL_NUM_STEPS are both specified.  One must be -1.\n"
                " NUM_UPDATES: {} TOTAL_NUM_STEPS: {}".format(
                    self.num_updates, self.total_num_steps
                )
            )

        if self.num_updates == -1 and self.total_num_steps == -1:
            raise RuntimeError(
                "One of NUM_UPDATES and TOTAL_NUM_STEPS must be specified.\n"
                " NUM_UPDATES: {} TOTAL_NUM_STEPS: {}".format(
                    self.num_updates, self.total_num_steps
                )
            )

        if self.num_checkpoints != -1 and self.checkpoint_interval != -1:
            raise RuntimeError(
                "NUM_CHECKPOINTS and CHECKPOINT_INTERVAL are both specified."
                "  One must be -1.\n"
                " NUM_CHECKPOINTS: {} CHECKPOINT_INTERVAL: {}".format(
                    self.num_checkpoints, self.checkpoint_interval 
                )
            )

        if self.num_checkpoints == -1 and self.checkpoint_interval == -1:
            raise RuntimeError(
                "One of NUM_CHECKPOINTS and CHECKPOINT_INTERVAL must be specified"
                " NUM_CHECKPOINTS: {} CHECKPOINT_INTERVAL: {}".format(
                    self.num_checkpoints, self.checkpoint_interval
                )
            )    

    def percent_done(self) -> float:
        if self.num_updates != -1:
            return self.num_updates_done / self.num_updates
        else:
            return self.num_steps_done / self.total_num_steps

    def is_done(self) -> bool:
        return self.percent_done() >= 1.0

    def should_checkpoint(self) -> bool:
        needs_checkpoint = False
        # use num_checkpoints
        if self.num_checkpoints != -1:
            # self._last_checkpoint_percent starts from -1
            # then will be saved and updated to the first percent done
            # then be updated each checkpoint_every
            # do not save at percentage 0
            checkpoint_every = 1 / self.num_checkpoints
            if (
                self._last_checkpoint_percent + checkpoint_every
                < self.percent_done()
            ):
                needs_checkpoint = True
                self._last_checkpoint_percent = self.percent_done()
        # use checkpoint_interval
        else:
            needs_checkpoint = (
                self.num_updates_done % self.checkpoint_interval
            ) == 0

        return needs_checkpoint

    def _should_save_resume_state(self) -> bool:
        return SAVE_STATE.is_set() or (
            (
                not self.config.get("preemption_save_state_batch_only")
                or is_slurm_batch_job()
            )
            and (
                (
                    int(self.num_updates_done + 1)
                    % int(self.config.get("preemption_save_resume_state_interval"))
                )
                == 0
            )
        )

    @property
    def flush_secs(self):
        return self._flush_secs

    @flush_secs.setter
    def flush_secs(self, value: int):
        self._flush_secs = value

    def train(self) -> None:
        raise NotImplementedError

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint. Trainer algorithms should
        implement this.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError

    @staticmethod
    def _pause_envs(
        envs_to_pause: List[int],
        envs: Union[VectorEnv, NavEnv],
        test_recurrent_hidden_states: Tensor,
        not_done_masks: Tensor,
        current_episode_reward: Tensor,
        prev_actions: Tensor,
        batch: Dict[str, Tensor],
        rgb_frames: Union[List[List[Any]], List[List[ndarray]]],
    ) -> Tuple[
        Union[VectorEnv, NavEnv],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Dict[str, Tensor],
        List[List[Any]],
    ]:
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # only keep the indices of non-paused envs, i.e. state_index (a list of indices)
            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                state_index
            ]
            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            # rgb frames are used for recording vidoes
            rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            test_recurrent_hidden_states,
            not_done_masks,
            current_episode_reward,
            prev_actions,
            batch,
            rgb_frames,
        )
