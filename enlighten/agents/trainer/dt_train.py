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

from enlighten.agents.models.decision_transformer import DecisionTransformer
from enlighten.agents.trainer.seq_trainer import SequenceTrainer
from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.agents.common.seed import set_seed_except_env_seed
from enlighten.agents.common.other import get_device
from enlighten.datasets.behavior_dataset import BehaviorDataset
from enlighten.envs.multi_nav_env import MultiNavEnv
from enlighten.agents.common.other import get_obs_channel_num

class DTTrainer(SequenceTrainer):
    def create_model(self):
        self.model = DecisionTransformer(
            obs_channel = get_obs_channel_num(self.config),
            obs_width = int(self.config.get("image_width")), 
            obs_height = int(self.config.get("image_height")),
            goal_dim=int(self.config.get("goal_dimension")),
            goal_form=self.config.get("goal_form"),
            act_num=int(self.config.get("action_number")),
            context_length=int(self.config.get('K')),
            max_ep_len=int(self.config.get("max_ep_len")), 
            pad_mode = str(self.config.get("pad_mode")), 
            hidden_size=int(self.config.get('embed_dim')), # parameters starting from here will be passed to gpt2
            n_layer=int(self.config.get('n_layer')),
            n_head=int(self.config.get('n_head')),
            n_inner=int(4*self.config.get('embed_dim')),
            activation_function=self.config.get('activation_function'),
            n_positions=1024,
            resid_pdrop=float(self.config.get('dropout')),
            attn_pdrop=float(self.config.get('dropout')),
        )

    # train for one step
    def train_one_step(self):
        # switch model to training mode
        self.model.train()
        
        # observation # (B,K,C,H,W)
        # action # (B,K)
        # goal # (B,K,goal_dim)
        # dtg # (B,K,1)
        # timestep # (B,K)
        # mask # (B,K)
        observations, actions, goals, timesteps, attention_mask, batch_shape = self.train_dataset.get_batch(self.batch_size)
        action_targets = torch.clone(actions)

        # [B,K, action_num+1]
        action_preds = self.model.forward(
            observations, actions, goals, timesteps, attention_mask=attention_mask,
        )

        # loss is computed over the whole sequence (K action tokens)
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1)[attention_mask.reshape(-1) > 0]

        #print(action_preds.size())  #[sum of seq_len, act_num]
        #print(action_targets.size()) #[sum seq_len]
        
        # loss is evaluated only on actions
        # action_targets are ground truth action indices (not one-hot vectors)
        loss =  F.cross_entropy(action_preds, action_targets)

        #print(loss) # a float number

        self.optimizer.zero_grad()
        # compute weight gradients
        loss.backward()
        # clip weight grad
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        # optimize for one step
        self.optimizer.step()

        return loss.detach().cpu().item()

    # self.config.get: config of wandb
    def train(self):
        # load behavior training data
        self.train_dataset = BehaviorDataset(self.config, self.device)

        # create model and move it to the correct device
        self.create_model()
        self.model = self.model.to(device=self.device)

        # print goal form
        print("==========> %s"%(self.config.get("goal_form")))

        # create optimizer: AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.get('learning_rate')),
            weight_decay=float(self.config.get('weight_decay')),
        )
        # Within warmup_steps, use <1*lr, then use lr
        warmup_steps = int(self.config.get('warmup_steps'))
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
            if self.config.get('eval_during_training') and self.eval_every_iterations > 0:
                if (iter+1) % self.eval_every_iterations == 0:
                    logs = self.eval_during_training(logs=logs, print_logs=True)
                    # add checkpoint index to evaluation logs
                    checkpoint_index = (iter+1) // self.eval_every_iterations
                    logs['checkpoints/eval_checkpoints'] = checkpoint_index
            
            # log to wandb
            if self.log_to_wandb:
                wandb.log(logs)
            
            # save checkpoint
            if (iter+1) % self.save_every_iterations == 0:
                self.save_checkpoint(checkpoint_number = int((iter+1) // self.save_every_iterations))
    
    # train for one iteration
    def train_one_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_action_losses, train_losses = [], []
        
        logs = dict()

        train_start = time.time()

        # switch model to training mode
        self.model.train()

        # train for num_steps
        for _ in range(num_steps):
            loss_dict = self.train_one_step()

            train_losses.append(loss_dict["loss"]) 
            train_action_losses.append(loss_dict["action_loss"]) 

            # step learning rate scheduler at each training step
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.start_time

        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_action_loss_mean'] = np.mean(train_action_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    
if __name__ == '__main__':
    trainer = DTTrainer(config_filename="imitation_learning_dt.yaml")
    trainer.train()
