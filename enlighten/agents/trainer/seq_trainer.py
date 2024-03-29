import gym
import numpy as np
import torch
import wandb
from torch.nn import functional as F

import argparse
import pickle
import random
import sys
import os
import datetime
import time

from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.agents.common.seed import set_seed_except_env_seed
from enlighten.agents.common.other import get_device
from enlighten.datasets.behavior_dataset import BehaviorDataset
from enlighten.envs.multi_nav_env import MultiNavEnv
from enlighten.agents.common.other import get_obs_channel_num
from enlighten.agents.evaluation.across_scene_single_env_evaluator import AcrossEnvEvaluatorSingle
from enlighten.agents.evaluation.across_scene_vec_env_evaluator import AcrossEnvEvaluatorVector 
from enlighten.datasets.image_dataset import ImageDataset


# train offline rl and imitation learning
class SequenceTrainer:
    def __init__(self, config_filename, resume, resume_experiment_name, resume_ckpt_index):
        assert config_filename is not None, "needs config file to initialize trainer"
        # get config
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)

        # seed everything except env
        self.seed = int(self.config.get("seed"))
        set_seed_except_env_seed(self.seed)

        # set device
        self.device = get_device(self.config)

        if self.config.get("eval_during_training"):
            # create evaluator during training
            # Note that only evaluator needs environment, offline training does not need
            if self.config.get("eval_use_vector_envs"):
                self.evaluator = AcrossEnvEvaluatorVector(eval_splits=list(self.config.get("eval_during_training_splits")),  
                config_filename=config_filename, device=self.device) 
            else:    
                self.evaluator = AcrossEnvEvaluatorSingle(eval_splits=list(self.config.get("eval_during_training_splits")),  
                config_filename=config_filename, device=self.device)

        # set resume flag
        self.resume = resume
        self.resume_experiment_name = resume_experiment_name
        self.resume_ckpt_index = resume_ckpt_index
        
        # set experiment name
        self.set_experiment_name()

        # init wandb
        self.log_to_wandb = self.config.get("log_to_wandb")
        if self.log_to_wandb:
            self.init_wandb()

    def set_experiment_name(self):
        self.project_name = self.config.get("algorithm_name").lower()

        self.group_name = self.config.get("experiment_name").lower()

        # experiment_name: seed-YearMonthDay-HourMiniteSecond
        # experiment name should be the same config run with different stochasticity
        now = datetime.datetime.now()
        #self.experiment_name = "s%d-"%(self.seed)+"-%s-"%(self.config.get("goal_form"))+now.strftime("%Y%m%d-%H%M%S").lower() 
        if self.resume:
            self.experiment_name = self.resume_experiment_name
        else:   
            self.experiment_name = "s%d-"%(self.seed)+now.strftime("%Y%m%d-%H%M%S").lower() 
        
    def init_wandb(self):
        # initialize this run under project xxx
        if self.resume:
            to_be_resume = True
        else:
            to_be_resume = None
        
        wandb.init(
            name=self.experiment_name,
            group=self.group_name,
            project=self.project_name,
            config=self.config,
            dir=os.path.join(root_path),
            #resume=to_be_resume
        )
        
        # Bug: cannot sync
        # if wandb.run.resumed:
        #     print("========> wandb run resumed")

    # resume checkpoint
    def resume_checkpoint(self):
        # load checkpoint
        folder_name = self.project_name + "-" + self.group_name + "-" + self.resume_experiment_name
        checkpoint_folder_path = os.path.join(checkpoints_path, folder_name)
        checkpoint_path = os.path.join(checkpoint_folder_path, f"ckpt_{self.resume_ckpt_index}.pth")
        checkpoint = torch.load(checkpoint_path)
        print("Loaded checkpoint at: %s"%(checkpoint_path))
        
        return checkpoint

    # save checkpoint
    def save_checkpoint(self, model, checkpoint_number, epoch_index):
        
        folder_name = self.project_name + "-" + self.group_name + "-" + self.experiment_name
        folder_path = os.path.join(checkpoints_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # save agent weights, config and optimizer
        if self.scheduler is None:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "epoch": epoch_index
            }
        else:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.config,
                "epoch": epoch_index
            }

        checkpoint_path = os.path.join(folder_path, f"ckpt_{checkpoint_number}.pth")
        torch.save(checkpoint, checkpoint_path)

        print(f"Checkpoint {checkpoint_number} saved.")
    
    # evaluate during training
    def eval_during_training(self, model, logs={}, print_logs=False):
        eval_start = time.time()

        # switch model to evaluation mode
        model.eval()
        
        # evaluate
        outputs, _ = self.evaluator.evaluate_over_datasets(model=model, sample=self.config.get("eval_during_training_sample"))
        for k, v in outputs.items():
            logs[f'evaluation/{k}'] = v

        
        logs['time/evaluation'] = time.time() - eval_start
        if print_logs:
            for k, v in logs.items():
                print(f'{k}: {v}')
        
        return logs


    
    

    
