import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys
import os
import datetime

from enlighten.agents.models.decision_transformer import DecisionTransformer
from enlighten.agents.trainer.seq_trainer import SequenceTrainer
from enlighten.agents.evaluation.evaluate_episodes import eval_episodes
from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.agents.common.seed import set_seed_except_env_seed
from enlighten.agents.common.other import get_device
from enlighten.datasets.behavior_dataset import BehaviorDataset
from enlighten.envs.multi_nav_env import MultiNavEnv
from enlighten.agents.common.other import get_obs_channel_num

class DTTrainer:
    def __init__(self, config_filename="imitation_learning.yaml"):
        assert config_filename is not None, "needs config file to initialize trainer"
        
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)

        # seed everything except env
        self.seed = int(self.config.get("seed"))
        set_seed_except_env_seed(self.seed)

        # create env, for evaluation, not for training
        self.env = MultiNavEnv(config_file=config_filename) 

        # set device
        self.device = get_device(self.config)
    
        # set experiment name
        self.set_experiment_name()

        # init wandb
        self.log_to_wandb = self.config.get("log_to_wandb")
        if self.log_to_wandb:
            self.init_wandb()
        
        # set save checkpoint parameters
        self.save_every_iterations = int(self.config.get("save_every_iterations"))
    
    def set_experiment_name(self):
        self.project_name = 'dt'.lower()

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
        train_dataset = BehaviorDataset(self.config, self.device)
        
        # create model and move it to the correct device
        model = DecisionTransformer(
            obs_channel = get_obs_channel_num(self.config),
            obs_width = int(self.config.get("image_width")), 
            obs_height = int(self.config.get("image_height")),
            goal_dim=int(self.config.get("goal_dimension")),
            goal_form=self.config.get("goal_form"),
            act_num=int(self.config.get("action_number")),
            context_length=int(self.config.get('K')),
            max_ep_len=int(self.config.get("max_ep_len")),  
            hidden_size=int(self.config.get('embed_dim')), # parameters starting from here will be passed to gpt2
            n_layer=int(self.config.get('n_layer')),
            n_head=int(self.config.get('n_head')),
            n_inner=int(4*self.config.get('embed_dim')),
            activation_function=self.config.get('activation_function'),
            n_positions=1024,
            resid_pdrop=float(self.config.get('dropout')),
            attn_pdrop=float(self.config.get('dropout')),
        )
        model = model.to(device=self.device)

        # print goal form
        print("==========> %s"%(self.config.get("goal_form")))

        # create optimizer: AdamW
        warmup_steps = int(self.config.get('warmup_steps'))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.get('learning_rate')),
            weight_decay=float(self.config.get('weight_decay')),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)
        )
        
        # create trainer
        batch_size = int(self.config.get('batch_size'))
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            train_dataset=train_dataset,
            scheduler=scheduler,
            #eval_fns=[eval_episodes()],
        )

        # train for max_iters iterations
        # each iteration includes num_steps_per_iter steps
        for iter in range(int(self.config.get('max_iters'))):
            outputs = trainer.train_one_iteration(num_steps=int(self.config.get('num_steps_per_iter')), iter_num=iter+1, print_logs=True)
            if self.log_to_wandb:
                wandb.log(outputs)
            
             # save checkpoint
            if (iter+1) % self.save_every_iterations == 0:
                self.save_checkpoint(trainer.model, checkpoint_number = int((iter+1) // self.save_every_iterations))
        
    # Save checkpoint
    def save_checkpoint(self, model, checkpoint_number):
        # only save agent weights
        checkpoint = model.state_dict()
        folder_name = self.project_name + "-" + self.group_name + "-" + self.experiment_name
        folder_path = os.path.join(checkpoints_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        checkpoint_path = os.path.join(folder_path, f"ckpt_{checkpoint_number}.pth")
        torch.save(checkpoint, checkpoint_path)

        print(f"Checkpoint {checkpoint_number} saved.")


if __name__ == '__main__':
    trainer = DTTrainer(config_filename="imitation_learning.yaml")
    trainer.train()
