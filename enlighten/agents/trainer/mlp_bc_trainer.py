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
from tqdm import tqdm


from enlighten.agents.models.mlp_policy_model import MLPPolicy
from enlighten.agents.trainer.seq_trainer import SequenceTrainer
from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.agents.common.seed import set_seed_except_env_seed
from enlighten.agents.common.other import get_device
from enlighten.datasets.behavior_dataset import BehaviorDataset
from enlighten.envs.multi_nav_env import MultiNavEnv
from enlighten.agents.common.other import get_obs_channel_num
from enlighten.datasets.image_dataset import ImageDataset

class MLPBCTrainer(SequenceTrainer):
    def __init__(self, config_filename, resume=False, resume_experiment_name=None, resume_ckpt_index=None):
        super(MLPBCTrainer, self).__init__(config_filename, resume, resume_experiment_name, resume_ckpt_index)

        # set evaluation interval
        self.eval_every_epochs = int(self.config.get("eval_every_epochs"))
        
        # set save checkpoint interval
        self.save_every_epochs = int(self.config.get("save_every_epochs"))


    def create_model(self):
        self.model = MLPPolicy(
            obs_channel = get_obs_channel_num(self.config),
            obs_width = int(self.config.get("image_width")), 
            obs_height = int(self.config.get("image_height")),
            goal_dim=int(self.config.get("goal_dimension")),
            goal_form=self.config.get("goal_form"),
            act_num=int(self.config.get("action_number")),
            obs_embedding_size=int(self.config.get('obs_embedding_size')), #512
            goal_embedding_size=int(self.config.get('goal_embedding_size')), #32
            hidden_size=int(self.config.get('hidden_size')),
            hidden_layer=int(self.config.get('hidden_layer')),
            state_form=self.config.get('state_form'),
            state_dimension=int(self.config.get('state_dimension'))
        )

    # train for one update
    def train_one_update(self):

        # switch model to training mode
        self.model.train()
        
        # (next)observations # (B,C,H,W)
        # action_targets # (B)
        # goals # (B,goal_dim)
        observations, goals, action_targets, rewards, next_observations, next_goals, dones, next_actions, optimal_action = self.train_dataset.get_transition_batch(self.batch_size)
        # forward model
        action_preds = self.model.forward(observations, goals)

        # action_preds: [B, action_num]
        # action_target are ground truth action indices (not one-hot vectors)
        action_loss =  F.cross_entropy(action_preds, action_targets)
            
        #print(loss) # a single float number

        self.optimizer.zero_grad()
        
        # compute weight gradients
        action_loss.backward()
        
        # optimize for one step
        self.optimizer.step()
        
        loss_dict = {}
        loss_dict["action_loss"] = action_loss.detach().cpu().item()

        # the number of updates ++
        self.updates_done += 1

        return loss_dict    

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
        
        self.scheduler = None

        # start training
        self.batch_size = int(self.config.get('batch_size'))
        self.start_time = time.time()

        # train for max_epochs
        # each epoch iterate over the whole training sets
        self.updates_done = 0
        for epoch in range(int(self.config.get('max_epochs'))):
            logs = self.train_one_epoch(epoch_num=epoch+1, print_logs=True)
            
            # evaluate during training
            if self.config.get('eval_during_training') and self.eval_every_epochs > 0:
                # do not eval at step 0
                if (epoch+1) % self.eval_every_epochs == 0:
                    logs = self.eval_during_training(model=self.model, logs=logs, print_logs=True)
                    # add eval point index to evaluation logs [index starting from 1]
                    eval_point_index = (epoch+1) // self.eval_every_epochs
                    # log evaluation checkpoint index, index starting from 1
                    logs['checkpoints/eval_checkpoints'] = eval_point_index
                    
            
            # log to wandb at every epoch
            if self.log_to_wandb:
                wandb.log(logs, step=epoch)
            
            
            # save checkpoint
            # do not save at epoch 0
            # checkpoint index starts from 0
            if (epoch+1) % self.save_every_epochs == 0:
                self.save_checkpoint(model=self.model, checkpoint_number = int((epoch+1) // self.save_every_epochs) - 1, epoch_index=epoch)
    
    # train for one epoch
    def train_one_epoch(self, epoch_num, print_logs=False):

        train_action_losses = []
        
        logs = dict()

        train_start = time.time()

        # switch model to training mode
        self.model.train()
        # shuffle training set
        self.train_dataset.shuffle_transition_dataset()
        
        # how many batches each epoch contains
        batch_num = self.train_dataset.get_transition_batch_num(self.batch_size)

        # train for n batches
        for _ in tqdm(range(batch_num)):
            loss_dict = self.train_one_update()

            # record losses
            train_action_losses.append(loss_dict["action_loss"])


        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.start_time

        logs['training/train_action_loss_mean'] = np.mean(train_action_losses)

        logs['training/train_action_loss_std'] = np.std(train_action_losses)

        logs['epoch'] = epoch_num
        logs['update'] = self.updates_done
        

        # print log at every epoch
        if print_logs:
            print('=' * 80)
            print(f'Epoch {epoch_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')
            
            print('=' * 80)

        return logs

    
if __name__ == '__main__':
    trainer = MLPBCTrainer(config_filename="imitation_learning_mlp_bc.yaml")
    trainer.train()
