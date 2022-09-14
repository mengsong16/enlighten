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
import time
from torch.nn import functional as F
from torch import nn

from enlighten.agents.models.rnn_seq_model import RNNSequenceModel
from enlighten.agents.trainer.seq_trainer import SequenceTrainer
from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.agents.common.seed import set_seed_except_env_seed
from enlighten.agents.common.other import get_device
from enlighten.datasets.behavior_dataset import BehaviorDataset
from enlighten.envs.multi_nav_env import MultiNavEnv
from enlighten.agents.common.other import get_obs_channel_num
from enlighten.datasets.image_dataset import ImageDataset

class RNNBCTrainer(SequenceTrainer):
    def __init__(self, config_filename):
        super(RNNBCTrainer, self).__init__(config_filename)

        # set evaluation interval
        self.eval_every_iterations = int(self.config.get("eval_every_iterations"))
        
        # set save checkpoint interval
        self.save_every_iterations = int(self.config.get("save_every_iterations"))

        # use value supervision during training or not
        self.supervise_value = self.config.get('supervise_value')
        print("==========> Supervise value: %r"%(self.supervise_value))

        # domain adaptation or not
        self.domain_adaptation = self.config.get('domain_adaptation')
        print("==========> Domain adaptation: %r"%(self.domain_adaptation))

    def create_model(self):
        self.model = RNNSequenceModel(
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

    # train for one step
    def train_one_step(self):

        # switch model to training mode
        self.model.train()
        
        # observations # (T,C,H,W)
        # prev_actions # (T)
        # action_targets # (T)
        # goals # (T,goal_dim)
        # batch_sizes # L
       
        observations, action_targets, prev_actions, goals, value_targets, batch_sizes, batch_shape = self.train_dataset.get_trajectory_batch(self.batch_size)
        if self.domain_adaptation == True:
            source_batch_size = observations.size(0)
            target_observations = self.target_domain_dataset.get_image_batch(batch_shape)
            target_batch_size = target_observations.size(0)
            # concat source and target observations
            observations = torch.cat((observations, target_observations), dim=0)
        
        # print(observations.size())
        # print(action_targets.size())
        # print(prev_actions.size())
        # print(goals.size())
        # print(batch_sizes.size())

        # create h0: [1, B, hidden_size]
        rnn_hidden_size = int(self.config.get('rnn_hidden_size'))
        h_0 = torch.zeros(1, self.batch_size, rnn_hidden_size, dtype=torch.float32, device=self.device) 

        # forward model
        if self.supervise_value == True and self.domain_adaptation == True:
            action_preds, value_preds, da_preds = self.model.forward(observations, prev_actions, goals, h_0, batch_sizes)
        elif self.supervise_value == False and self.domain_adaptation == True:
            action_preds, da_preds = self.model.forward(observations, prev_actions, goals, h_0, batch_sizes)
        elif self.supervise_value == True and self.domain_adaptation == False:
            action_preds, value_preds = self.model.forward(observations, prev_actions, goals, h_0, batch_sizes)
        else:
            action_preds = self.model.forward(observations, prev_actions, goals, h_0, batch_sizes)

        # action loss is computed over the whole sequence
        # action_preds: [T, action_num]
        # action_target are ground truth action indices (not one-hot vectors)
        loss = 0
        action_loss =  F.cross_entropy(action_preds, action_targets)
        loss += action_loss
        # + value prediction
        if self.supervise_value == True:
            value_loss = torch.mean((value_targets - value_preds)**2)
            loss += value_loss
        if self.domain_adaptation == True:
            # source: 1, target: 0
            domain_labels = torch.tensor([[1], ] * source_batch_size + [[0], ] * target_batch_size, device=self.device, dtype=torch.float32)
            adv_loss = F.binary_cross_entropy_with_logits(da_preds, domain_labels)
            loss += adv_loss
            
        #print(loss) # a single float number

        self.optimizer.zero_grad()
        # compute weight gradients
        loss.backward()
        
        # optimize for one step
        self.optimizer.step()
        
        loss_dict = {}
        loss_dict["loss"] = loss.detach().cpu().item()
        loss_dict["action_loss"] = action_loss.detach().cpu().item()

        if self.supervise_value == True:
            loss_dict["value_loss"] = value_loss.detach().cpu().item()
        if self.domain_adaptation == True:
            loss_dict["adv_loss"] = adv_loss.detach().cpu().item() 
        
        return loss_dict    

    # self.config.get: config of wandb
    def train(self):
        # load behavior training data
        self.train_dataset = BehaviorDataset(self.config, self.device)
        
        if self.domain_adaptation:
            self.target_domain_dataset = ImageDataset(self.config)

        # create model and move it to the correct device
        self.create_model()
        self.model = self.model.to(device=self.device)

        # print goal form
        print("goal form ==========> %s"%(self.config.get("goal_form")))

        # create optimizer: AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.get('learning_rate')),
            weight_decay=float(self.config.get('weight_decay')),
        )
        self.scheduler = None
        
        # start training
        self.batch_size = int(self.config.get('batch_size'))
        self.start_time = time.time()

        # train for max_iters iterations
        # each iteration includes num_steps_per_iter steps
        for iter in range(int(self.config.get('max_iters'))):
            logs = self.train_one_iteration(num_steps=int(self.config.get('num_steps_per_iter')), iter_num=iter+1, print_logs=True)
            
            # evaluate
            if self.config.get('eval_during_training') and self.eval_every_iterations > 0:
                # do not eval at step 0
                if (iter+1) % self.eval_every_iterations == 0:
                    logs = self.eval_during_training(logs=logs, print_logs=True)
                    # add eval point index to evaluation logs, index starting from 1
                    eval_point_index = (iter+1) // self.eval_every_iterations
                    logs['checkpoints/eval_checkpoints'] = eval_point_index
                    
            
            # log current logs to wandb
            if self.log_to_wandb:
                wandb.log(logs)
            
            
            # save checkpoint
            # do not save at step 0
            # checkpoint index starts from 0
            if (iter+1) % self.save_every_iterations == 0:
                self.save_checkpoint(checkpoint_number = int((iter+1) // self.save_every_iterations) - 1)
    
    # train for one iteration
    def train_one_iteration(self, num_steps, iter_num, print_logs=False):

        train_action_losses, train_losses = [], []
        if self.supervise_value:
            train_value_losses = []
        
        if self.domain_adaptation:
            train_da_losses = []

        logs = dict()

        train_start = time.time()

        # switch model to training mode
        self.model.train()

        # train for num_steps
        for _ in range(num_steps):
            loss_dict = self.train_one_step()

            train_losses.append(loss_dict["loss"]) 
            train_action_losses.append(loss_dict["action_loss"])

            if self.supervise_value:
                train_value_losses.append(loss_dict["value_loss"])
            if self.domain_adaptation:
                train_da_losses.append(loss_dict["adv_loss"]) 

            # step learning rate scheduler at each training step
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.start_time

        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_action_loss_mean'] = np.mean(train_action_losses)

        if self.supervise_value:
            logs['training/train_value_loss_mean'] = np.mean(train_value_losses)
        if self.domain_adaptation:
            logs['training/train_da_loss_mean'] = np.mean(train_da_losses)

        logs['training/train_loss_std'] = np.std(train_losses)

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    
if __name__ == '__main__':
    trainer = RNNBCTrainer(config_filename="imitation_learning_rnn_bc.yaml")
    trainer.train()
