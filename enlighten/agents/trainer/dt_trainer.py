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
    # resume_ckpt_index index starting from 0
    def __init__(self, config_filename, resume=False, resume_experiment_name=None, resume_ckpt_index=None):
        super(DTTrainer, self).__init__(config_filename, resume, resume_experiment_name, resume_ckpt_index)

        # set evaluation interval
        self.eval_every_iterations = int(self.config.get("eval_every_iterations"))
        
        # set save checkpoint interval
        self.save_every_iterations = int(self.config.get("save_every_iterations"))

    def create_model(self):
        self.model = DecisionTransformer(
            obs_channel = get_obs_channel_num(self.config),
            obs_width = int(self.config.get("image_width")), 
            obs_height = int(self.config.get("image_height")),
            goal_dim=int(self.config.get("goal_dimension")),
            goal_form=self.config.get("goal_form"),
            act_num=int(self.config.get("action_number")),
            context_length=int(self.config.get('K')),
            max_ep_len=int(self.config.get("dt_max_ep_len")), 
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
        observations, actions, goals, timesteps, attention_mask, batch_shape = self.train_dataset.get_trajectory_batch(self.batch_size)
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
        #print("goal form ==========> %s"%(self.config.get("goal_form")))

        # create optimizer: 
        # AdamW (Adam with weight decay)
        if self.config.get("optimizer") == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(self.config.get('learning_rate')),
                weight_decay=float(self.config.get('weight_decay')),
            )
        elif self.config.get("optimizer") == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=float(self.config.get('learning_rate'))
            )
        else:
            print("Error: unknown optimizer: %s"%(self.config.get("optimizer")))
            exit()
        
        print("======> created optimizer: %s"%(self.config.get("optimizer")))
        
        # Within warmup_steps, use <1*lr, then use lr
        warmup_steps = int(self.config.get('warmup_steps'))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)
        )

        # resume from the checkpoint
        if self.resume:
            checkpoint = self.resume_checkpoint()
            # resume model, optimizer, scheduler
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if "epoch" in checkpoint.keys():
                start_iter = checkpoint['epoch'] + 1
            else:
                start_iter = (self.resume_ckpt_index + 1) * self.save_every_iterations
            print("=======> Will resume training starting from iteration index %d"%(start_iter))
        else:
            start_iter = 0
        
        # start training
        self.batch_size = int(self.config.get('batch_size'))
        self.start_time = time.time()

        print("======> Start training from epoch %d to epoch %d"%(start_iter, int(self.config.get('max_iters'))-1))

        # train for max_iters iterations
        # each iteration includes num_steps_per_iter steps
        for iter in range(start_iter, int(self.config.get('max_iters'))):
            logs = self.train_one_iteration(num_steps=int(self.config.get('num_steps_per_iter')), iter_num=iter+1, print_logs=True)
            
            # evaluate
            if self.config.get('eval_during_training') and self.eval_every_iterations > 0:
                if (iter+1) % self.eval_every_iterations == 0:
                    logs = self.eval_during_training(model=self.model, logs=logs, print_logs=True)
                    # add checkpoint index to evaluation logs
                    checkpoint_index = (iter+1) // self.eval_every_iterations
                    logs['checkpoints/eval_checkpoints'] = checkpoint_index
            
            # log to wandb at every iteration
            if self.log_to_wandb:
                wandb.log(logs, step=iter)
            
            # save checkpoint
            # do not save at iter 0
            # iter starts from 0
            # checkpoint index starts from 1
            if (iter+1) % self.save_every_iterations == 0:
                self.save_checkpoint(model=self.model, checkpoint_number = int((iter+1) // self.save_every_iterations), epoch_index=iter)
    
    # train for one iteration
    # iter_num: the number of iterations that will be done (starts from 1)
    def train_one_iteration(self, num_steps, iter_num, print_logs=False):

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
