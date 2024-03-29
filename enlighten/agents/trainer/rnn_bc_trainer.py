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
    # resume_ckpt_index index starting from 0
    def __init__(self, config_filename, resume=False, resume_experiment_name=None, resume_ckpt_index=None):
        super(RNNBCTrainer, self).__init__(config_filename, resume, resume_experiment_name, resume_ckpt_index)

        # set evaluation interval
        self.eval_every_epochs = int(self.config.get("eval_every_epochs"))
        
        # set save checkpoint interval
        self.save_every_epochs = int(self.config.get("save_every_epochs"))

        # use value supervision during training or not
        self.supervise_value = self.config.get('supervise_value')
        print("==========> Supervise value: %r"%(self.supervise_value))

        # domain adaptation or not
        self.domain_adaptation = self.config.get('domain_adaptation')
        print("==========> Domain adaptation: %r"%(self.domain_adaptation))
    
        # action type
        self.action_type = self.config.get("action_type", "cartesian")
        print("=========> Action type: %s"%(self.action_type))
        
        # action number 
        if self.action_type == "polar":
            self.action_number = 1 + int(360 // int(self.config.get("rotate_resolution")))
        else:
            self.action_number = int(self.config.get("action_number"))
        print("=========> Action number: %d"%(self.action_number))


    # suport cartesian or polar action space
    def create_model(self):
        self.model = RNNSequenceModel(
            obs_channel = get_obs_channel_num(self.config),
            obs_width = int(self.config.get("image_width")), 
            obs_height = int(self.config.get("image_height")),
            goal_dim=int(self.config.get("goal_dimension")),
            goal_form=self.config.get("goal_form"),
            act_num=self.action_number,  
            rnn_hidden_size=int(self.config.get('rnn_hidden_size')), 
            obs_embedding_size=int(self.config.get('obs_embedding_size')), #512
            goal_embedding_size=int(self.config.get('goal_embedding_size')), #32
            act_embedding_size=int(self.config.get('act_embedding_size')), #32
            rnn_type=self.config.get('rnn_type'),
            supervise_value=self.config.get('supervise_value'),
            domain_adaptation=self.config.get('domain_adaptation'),
            temperature=float(self.config.get('temperature', 1.0))
        )

    # train for one update
    def train_one_update(self):

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
        
        # the number of updates ++
        self.updates_done += 1

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
        #print("goal form ==========> %s"%(self.config.get("goal_form")))

        # create optimizer: AdamW
        # self.optimizer = torch.optim.AdamW(
        #     self.model.parameters(),
        #     lr=float(self.config.get('learning_rate')),
        #     weight_decay=float(self.config.get('weight_decay')),
        # )
        # self.scheduler = None

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
        
        # resume from the checkpoint
        if self.resume:
            checkpoint = self.resume_checkpoint()
            # resume model, optimizer, scheduler
            self.model.load_state_dict(checkpoint['model_state_dict'])
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
                    logs = self.eval_during_training(model=self.model, logs=logs, print_logs=True)
                    # add eval point index to evaluation logs, index starting from 1
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
    # epoch_num: the number of epochs that will be done (starts from 1)
    def train_one_epoch(self, epoch_num, print_logs=False):

        train_action_losses, train_losses = [], []
        if self.supervise_value:
            train_value_losses = []
        
        if self.domain_adaptation:
            train_da_losses = []

        logs = dict()

        train_start = time.time()

        # switch model to training mode
        self.model.train()

        # shuffle training set
        self.train_dataset.shuffle_trajectory_dataset()

        # how many batches each epoch contains: 500
        batch_num = self.train_dataset.get_trajectory_batch_num(self.batch_size)

        # train for one epoch
        for _ in tqdm(range(batch_num)):
            loss_dict = self.train_one_update()

            # record losses
            train_losses.append(loss_dict["loss"]) 
            train_action_losses.append(loss_dict["action_loss"])

            if self.supervise_value:
                train_value_losses.append(loss_dict["value_loss"])
            if self.domain_adaptation:
                train_da_losses.append(loss_dict["adv_loss"]) 

            # step learning rate scheduler at each training step
            # if self.scheduler is not None:
            #     self.scheduler.step()

        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.start_time

        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_action_loss_mean'] = np.mean(train_action_losses)

        logs['epoch'] = epoch_num
        logs['update'] = self.updates_done

        if self.supervise_value:
            logs['training/train_value_loss_mean'] = np.mean(train_value_losses)
        if self.domain_adaptation:
            logs['training/train_da_loss_mean'] = np.mean(train_da_losses)

        logs['training/train_loss_std'] = np.std(train_losses)

        if print_logs:
            print('=' * 80)
            print(f'Epoch {epoch_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    
if __name__ == '__main__':
    trainer = RNNBCTrainer(
        config_filename="imitation_learning_rnn_bc.yaml")

    # trainer = RNNBCTrainer(
    #     config_filename="imitation_learning_rnn_bc.yaml",
    #     resume=True,
    #     resume_experiment_name="s1-20221007-021313",
    #     resume_ckpt_index=12)

    trainer.train()
