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
from enlighten.agents.evaluation.evaluate_episodes import MultiEnvEvaluator


# train seq2seq imitation learning
class SequenceTrainer:
    def __init__(self, config_filename):
        assert config_filename is not None, "needs config file to initialize trainer"
        
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)

        # seed everything except env
        self.seed = int(self.config.get("seed"))
        set_seed_except_env_seed(self.seed)

        # set device
        self.device = get_device(self.config)

        # create evaluator
        # Note that only evaluator needs environment, offline training does not need
        self.evaluator = MultiEnvEvaluator(eval_splits=list(self.config.get("eval_during_training_splits")),  
            config_filename=config_filename, device=self.device)

    
        # set experiment name
        self.set_experiment_name()

        # init wandb
        self.log_to_wandb = self.config.get("log_to_wandb")
        if self.log_to_wandb:
            self.init_wandb()
        
        # set evaluation 
        self.eval_every_iterations = int(self.config.get("eval_every_iterations"))
        
        # set save checkpoint parameters
        self.save_every_iterations = int(self.config.get("save_every_iterations"))
    
    def set_experiment_name(self):
        self.project_name = self.config.get("algorithm_name").lower()

        self.group_name = self.config.get("experiment_name").lower()

        # experiment_name: seed-YearMonthDay-HourMiniteSecond
        # experiment name should be the same config run with different stochasticity
        now = datetime.datetime.now()
        #self.experiment_name = "s%d-"%(self.seed)+"-%s-"%(self.config.get("goal_form"))+now.strftime("%Y%m%d-%H%M%S").lower() 
        self.experiment_name = "s%d-"%(self.seed)+now.strftime("%Y%m%d-%H%M%S").lower() 
        
    def init_wandb(self):
        # initialize this run under project xxx
        wandb.init(
            name=self.experiment_name,
            group=self.group_name,
            project=self.project_name,
            config=self.config,
            dir=os.path.join(root_path)
        )

    
    # self.config.get: config of wandb
    def train(self):
        # load training data
        self.train_dataset = BehaviorDataset(self.config, self.device)
        
        # create model and move it to the correct device
        self.create_model()
        self.model = self.model.to(device=self.device)

        # print goal form
        print("==========> %s"%(self.config.get("goal_form")))

        # create optimizer: AdamW
        warmup_steps = int(self.config.get('warmup_steps'))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.get('learning_rate')),
            weight_decay=float(self.config.get('weight_decay')),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)
        )
        
        # start training
        self.batch_size = int(self.config.get('batch_size'))
        self.start_time = time.time()

        # train for max_iters iterations
        # each iteration includes num_steps_per_iter steps
        for iter in range(int(self.config.get('max_iters'))):
            logs = self.train_one_iteration(num_steps=int(self.config.get('num_steps_per_iter')), iter_num=iter+1, print_logs=True)
            
            # evaluate
            if self.eval_every_iterations > 0:
                if (iter+1) % self.eval_every_iterations == 0:
                    self.eval_during_training(logs=logs, print_logs=True)
                
            if self.log_to_wandb:
                wandb.log(logs)
            
            # save checkpoint
            if (iter+1) % self.save_every_iterations == 0:
                self.save_checkpoint(checkpoint_number = int((iter+1) // self.save_every_iterations))
    
    # train for one iteration
    def train_one_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        # switch model to training mode
        self.model.train()

        # train for num_steps
        for _ in range(num_steps):
            train_loss = self.train_one_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.start_time
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    
    # save checkpoint
    def save_checkpoint(self, checkpoint_number):
        # only save agent weights
        checkpoint = self.model.state_dict()
        folder_name = self.project_name + "-" + self.group_name + "-" + self.experiment_name
        folder_path = os.path.join(checkpoints_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        checkpoint_path = os.path.join(folder_path, f"ckpt_{checkpoint_number}.pth")
        torch.save(checkpoint, checkpoint_path)

        print(f"Checkpoint {checkpoint_number} saved.")
    
    # evaluate during training
    def eval_during_training(self, logs={}, print_logs=False):
        eval_start = time.time()

        # switch model to evaluation mode
        self.model.eval()
        
        # evaluate
        outputs = self.evaluator.evaluate_over_datasets(model=self.model, sample=self.config.get("eval_during_training_sample"))
        for k, v in outputs.items():
            logs[f'evaluation/{k}'] = v
        
        logs['time/evaluation'] = time.time() - eval_start
        if print_logs:
            for k, v in logs.items():
                print(f'{k}: {v}')
        
        return logs


    
    

    
