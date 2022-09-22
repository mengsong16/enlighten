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


from enlighten.agents.models.q_network import QNetwork
from enlighten.agents.trainer.seq_trainer import SequenceTrainer
from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.agents.common.seed import set_seed_except_env_seed
from enlighten.agents.common.other import get_device
from enlighten.datasets.behavior_dataset import BehaviorDataset
from enlighten.envs.multi_nav_env import MultiNavEnv
from enlighten.agents.common.other import get_obs_channel_num
from enlighten.datasets.image_dataset import ImageDataset

class DQNTrainer(SequenceTrainer):
    def __init__(self, config_filename):
        super(DQNTrainer, self).__init__(config_filename)

        # set evaluation interval
        self.eval_every_epochs = int(self.config.get("eval_every_epochs"))
        
        # set save checkpoint interval
        self.save_every_epochs = int(self.config.get("save_every_epochs"))

        # gamma
        self.gamma = float(self.config.get("gamma"))

        # target q parameters
        self.target_update_every_updates = int(self.config.get("target_update_every_updates"))
        self.soft_target_tau = float(self.config.get("soft_target_tau"))


    def create_model(self):
        self.q_network = QNetwork(
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

        self.target_q_network = QNetwork(
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

        # load the weights into the target networks
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    # polyak update
    # tau = 1: 100% copy from source to target
    def soft_update_from_to(self, source, target, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
        
    # train for one update
    def train_one_update(self):

        # switch model mode
        self.q_network.train()
        self.target_q_network.eval()
        
        # (next)observations # (B,C,H,W)
        # actions # (B)
        # rewards # (B)
        # goals # (B,goal_dim)
        # dones # (B)
        observations, goals, actions, rewards, next_observations, next_goals, dones, next_actions = self.train_dataset.get_transition_batch(self.batch_size)
        
        # compute target Q
        with torch.no_grad():
            # Q_targets_next, _ = torch.max(self.target_q_network(next_observations, next_goals).detach(), 1) #[B]
            # Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones.int())) # dones: bool to int
            
            Q_targets_next = self.target_q_network(next_observations, next_goals).detach()
            Q_target_best_next = torch.gather(Q_targets_next,
                                    dim=1,
                                    index=next_actions.long().unsqueeze(1)).squeeze(1) # [B]
            Q_targets = rewards + (self.gamma * Q_target_best_next * (1 - dones.int()))
            Q_targets = Q_targets.detach() #[B]
        
        # compute predicted Q
        Q_predicted = torch.gather(self.q_network(observations, goals),
                                    dim=1,
                                    index=actions.long().unsqueeze(1)).squeeze(1) # [B]

        # compute Q loss
        q_loss = F.mse_loss(Q_predicted, Q_targets) # a single float number

        # optimize Q network
        self.optimizer.zero_grad()
    
        q_loss.backward()
        
        self.optimizer.step()
        
        # record q loss
        loss_dict = {}
        loss_dict["q_loss"] = q_loss.detach().cpu().item()

        # soft update target Q network (update when total updates == 0)
        if self.updates_done % self.target_update_every_updates == 0:
            self.soft_update_from_to(
                self.q_network, self.target_q_network, self.soft_target_tau)

        # the number of updates ++
        self.updates_done += 1

        return loss_dict    

    # self.config.get: config of wandb
    def train(self):
        # load behavior training data
        self.train_dataset = BehaviorDataset(self.config, self.device)
        
        
        # create model and move it to the correct device
        self.create_model()
        self.q_network = self.q_network.to(device=self.device)
        self.target_q_network = self.target_q_network.to(device=self.device)

        # print goal form
        print("goal form ==========> %s"%(self.config.get("goal_form")))

        # create optimizer: Adam
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=float(self.config.get('learning_rate'))
        )
        
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
                    logs = self.eval_during_training(model=self.q_network, logs=logs, print_logs=True)
                    # add eval point index to evaluation logs [index starting from 1]
                    eval_point_index = (epoch+1) // self.eval_every_epochs
                    # log evaluation checkpoint index, index starting from 1
                    logs['checkpoints/eval_checkpoints'] = eval_point_index
                    
            
            # log to wandb at every epoch
            if self.log_to_wandb:
                wandb.log(logs)
            
            
            # save checkpoint
            # do not save at step 0
            # checkpoint index starts from 0
            if (epoch+1) % self.save_every_epochs == 0:
                self.save_checkpoint(model=self.q_network, checkpoint_number = int((epoch+1) // self.save_every_epochs) - 1)
    
    # train for one epoch
    def train_one_epoch(self, epoch_num, print_logs=False):

        train_q_losses = []
        
        logs = dict()

        train_start = time.time()

        # switch model to training mode
        self.q_network.train()
        # shuffle training set
        self.train_dataset.shuffle_transition_dataset()
        
        # how many batches each epoch contains: 239
        batch_num = self.train_dataset.get_transition_batch_num(self.batch_size)

        # train for n batches
        for _ in tqdm(range(batch_num)):
            loss_dict = self.train_one_update()

            # record losses
            train_q_losses.append(loss_dict["q_loss"]) 

        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.start_time

        logs['training/train_q_loss_mean'] = np.mean(train_q_losses)

        logs['training/train_q_loss_std'] = np.std(train_q_losses)

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
    trainer = DQNTrainer(config_filename="imitation_learning_dqn.yaml")
    trainer.train()
