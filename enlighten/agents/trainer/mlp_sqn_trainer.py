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

class MLPSQNTrainer(SequenceTrainer):
    # resume_ckpt_index index starting from 0
    def __init__(self, config_filename, resume=False, resume_experiment_name=None, resume_ckpt_index=None):
        super(MLPSQNTrainer, self).__init__(config_filename, resume, resume_experiment_name, resume_ckpt_index)

        # set evaluation interval
        self.eval_every_epochs = int(self.config.get("eval_every_epochs"))
        
        # set save checkpoint interval
        self.save_every_epochs = int(self.config.get("save_every_epochs"))

        # reward type
        self.reward_type = self.config.get("reward_type")
        
        # action type
        self.action_type = self.config.get("action_type", "cartesian")
        print("=========> Action type: %s"%(self.action_type))
        assert self.action_type == "polar", "Supevised Q learning assumes the action type as polar"


    def create_model(self):
        if self.action_type == "polar":
            self.action_number = 37
        else:
            self.action_number = int(self.config.get("action_number"))

        self.q_network = QNetwork(
            obs_channel = get_obs_channel_num(self.config),
            obs_width = int(self.config.get("image_width")), 
            obs_height = int(self.config.get("image_height")),
            goal_dim=int(self.config.get("goal_dimension")),
            goal_form=self.config.get("goal_form"),
            act_num=self.action_number,
            obs_embedding_size=int(self.config.get('obs_embedding_size')), #512
            goal_embedding_size=int(self.config.get('goal_embedding_size')), #32
            hidden_size=int(self.config.get('hidden_size')),
            hidden_layer=int(self.config.get('hidden_layer')),
            state_form=self.config.get('state_form'),
            state_dimension=int(self.config.get('state_dimension'))
        )

    # train for one update
    def train_one_update(self):

        # switch model mode
        self.q_network.train()
        
        # observations # (B,C,H,W)
        # goals # (B,goal_dim)
        # qs # (B, action_number)
        # actions # (B)
        # rewards # (B)
        # dones # (B)
        observations, goals, q_groundtruths, actions, rewards, dones = self.train_dataset.get_transition_batch(self.batch_size)

        # compute predicted Q
        # (B, action_number)
        Q_output = self.q_network(observations, goals)
        
        # compute Q loss
        q_loss = F.mse_loss(Q_output, q_groundtruths) # a single float number
        
        
        # optimize Q network
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()
        
        # record losses
        loss_dict = {}
        loss_dict["q_loss"] = q_loss.detach().cpu().item()

        # count how many actions are predicted correctly
        # with torch.no_grad():
        #     predicted_actions = torch.argmax(Q_output.detach(), dim=1)
        #     compare_actions = torch.eq(predicted_actions, actions)
        #     action_pred_correct = torch.sum(compare_actions.int())
        #     loss_dict["correct_action_num"] = action_pred_correct.detach().cpu().item()

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
        

        # create optimizer: 
        if self.config.get("optimizer") == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.q_network.parameters(),
                lr=float(self.config.get('learning_rate')),
                weight_decay=float(self.config.get('weight_decay')),
            )
        elif self.config.get("optimizer") == "Adam":
            self.optimizer = torch.optim.Adam(
                self.q_network.parameters(),
                lr=float(self.config.get('learning_rate'))
            )
        else:
            print("Error: unknown optimizer: %s"%(self.config.get("optimizer")))
            exit()
        
        print("======> created optimizer: %s"%(self.config.get("optimizer")))
        self.scheduler = None

        # resume from the checkpoint
        if self.resume:
            checkpoint = self.resume_checkpoint()
            # resume model, optimizer, scheduler
            self.q_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if "epoch" in checkpoint.keys():
                start_epoch = checkpoint['epoch'] + 1
            else:
                start_epoch = (self.resume_ckpt_index + 1) * self.save_every_epochs
            print("=======> Will resume training starting from epoch index %d"%(start_epoch))
        else:
            start_epoch = 0
        
        # start training
        self.batch_size = int(self.config.get('batch_size'))
        self.start_time = time.time()

        print("======> Start training from epoch %d to epoch %d"%(start_epoch, int(self.config.get('max_epochs'))-1))

        # train for max_epochs
        # each epoch iterate over the whole training sets
    
        self.updates_done = 0
        for epoch in range(start_epoch, int(self.config.get('max_epochs'))):
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
                wandb.log(logs, step=epoch)
            
            
            # save checkpoint
            # do not save at epoch 0
            # checkpoint index starts from 0
            if (epoch+1) % self.save_every_epochs == 0:
                self.save_checkpoint(model=self.q_network, checkpoint_number = int((epoch+1) // self.save_every_epochs) - 1, epoch_index=epoch)
    
    # train for one epoch
    # epoch_num: the number of epochs that will be done (starts from 1)
    def train_one_epoch(self, epoch_num, print_logs=False):

        train_q_losses = []
        #train_action_correct_nums = []
        
        logs = dict()

        train_start = time.time()

        # switch model to training mode
        self.q_network.train()
        # shuffle training set
        self.train_dataset.shuffle_transition_dataset()
        
        # how many batches each epoch contains: 239
        batch_num = self.train_dataset.get_transition_batch_num(self.batch_size)

        # train for one epoch
        for _ in tqdm(range(batch_num)):
                
            loss_dict = self.train_one_update()

            # record losses
            train_q_losses.append(loss_dict["q_loss"]) 
            #train_action_correct_nums.append(loss_dict["correct_action_num"])
                

        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.start_time

        logs['training/q_loss_mean'] = np.mean(train_q_losses)
        logs['training/q_loss_std'] = np.std(train_q_losses)

        # logs['training/action_accuracy'] = np.sum(train_action_correct_nums) / float(self.train_dataset.total_transition_num())


        logs['epoch'] = epoch_num
        logs['update'] = self.updates_done
        

        # print log at the end of every epoch
        if print_logs:
            print('=' * 80)
            print(f'Epoch {epoch_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')
            
            print('=' * 80)

        return logs

    
if __name__ == '__main__':
    # trainer = MLPSQNTrainer(
    #     config_filename="imitation_learning_mlp_sqn.yaml", 
    #     resume=True,
    #     resume_experiment_name="s1-20221005-174833",
    #     resume_ckpt_index=10)
    
    trainer = MLPSQNTrainer(
        config_filename="imitation_learning_mlp_sqn.yaml")
    
    trainer.train()
