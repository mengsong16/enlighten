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
    # resume_ckpt_index index starting from 0
    def __init__(self, config_filename, resume=False, resume_experiment_name=None, resume_ckpt_index=None):
        super(DQNTrainer, self).__init__(config_filename, resume, resume_experiment_name, resume_ckpt_index)

        # set evaluation interval
        self.eval_every_epochs = int(self.config.get("eval_every_epochs"))
        
        # set save checkpoint interval
        self.save_every_epochs = int(self.config.get("save_every_epochs"))

        # gamma
        self.gamma = float(self.config.get("gamma"))

        
        # target q parameters
        self.target_update_every_updates = int(self.config.get("target_update_every_updates"))
        self.soft_target_tau = float(self.config.get("soft_target_tau"))

        # q learning type
        self.q_learning_type = self.config.get("q_learning_type")
        print("q learning type =====> %s"%(self.q_learning_type))


        # with bc loss
        self.with_bc_loss = self.config.get("with_bc_loss")
        print("with bc loss =====> %s"%(self.with_bc_loss))

        if self.with_bc_loss:
            self.q_bc_weight = float(self.config.get("q_bc_weight"))
        
        # reward type
        self.reward_type = self.config.get("reward_type")

        # reward scale
        if self.reward_type == "minus_one_zero":
            self.negative_reward_scale = float(self.config.get("negative_reward_scale", 1.0))

        # number of actions
        self.action_number = int(self.config.get("action_number"))


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
    def train_one_update_others(self):

        # switch model mode
        self.q_network.train()
        self.target_q_network.train()
        
        # (next)observations # (B,C,H,W)
        # actions # (B)
        # rewards # (B)
        # goals # (B,goal_dim)
        # dones # (B)
        observations, goals, actions, rewards, next_observations, next_goals, dones, next_actions, optimal_action = self.train_dataset.get_transition_batch(self.batch_size)


        # compute target Q
        with torch.no_grad():
            if self.q_learning_type == "dqn":
                Q_target_best_next, _ = torch.max(self.target_q_network(next_observations, next_goals).detach(), 1) #[B] 
            elif self.q_learning_type == "double_q":
                next_best_actions = torch.argmax(self.q_network(next_observations, next_goals), dim=1)
                Q_targets_next = self.target_q_network(next_observations, next_goals)
                Q_target_best_next = torch.gather(Q_targets_next,
                                    dim=1,
                                    index=next_best_actions.long().unsqueeze(1)).squeeze(1).detach()
                
            elif self.q_learning_type == "no_max":
                Q_targets_next = self.target_q_network(next_observations, next_goals).detach()
                Q_target_best_next = torch.gather(Q_targets_next,
                                        dim=1,
                                        index=next_actions.long().unsqueeze(1)).squeeze(1) # [B]
                
            else:
                print("Error: unimplemented q learning type: %s"%(self.q_learning_type))
                exit()
            
            Q_targets = rewards + self.gamma * Q_target_best_next * (1.0 - dones.float()) # dones: bool to float
            Q_targets = Q_targets.detach() #[B]
        
        # compute predicted Q
        Q_output = self.q_network(observations, goals)
        Q_predicted = torch.gather(Q_output,
                                    dim=1,
                                    index=actions.long().unsqueeze(1)).squeeze(1) # [B]

        
        # compute Q loss
        q_loss = F.mse_loss(Q_predicted, Q_targets) # a single float number
        
        # compute lambda
        if self.with_bc_loss:
            #lmbda = self.q_bc_weight / Q_predicted.abs().mean().detach()
            lmbda = self.q_bc_weight
        else:
            lmbda = 1.0
        
        # total loss
        loss = lmbda * q_loss

        if self.with_bc_loss:
            action_preds = self.q_network.forward(observations, goals)
            action_loss =  F.cross_entropy(action_preds, actions)
            loss += action_loss
        
        # optimize Q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # record losses
        loss_dict = {}
        loss_dict["q_loss"] = q_loss.detach().cpu().item()
        loss_dict["bc_loss"] = action_loss.detach().cpu().item()
        loss_dict["loss"] = loss.detach().cpu().item()

        # count how many actions are predicted correctly
        with torch.no_grad():
            predicted_actions = torch.argmax(Q_output.detach(), dim=1)
            compare_actions = torch.eq(predicted_actions, actions)
            action_pred_correct = torch.sum(compare_actions.int())
            loss_dict["correct_action_num"] = action_pred_correct.detach().cpu().item()

        # soft update target Q network (update when total updates == 0)
        if self.updates_done % self.target_update_every_updates == 0:
            self.soft_update_from_to(
                self.q_network, self.target_q_network, self.soft_target_tau)

        # the number of updates ++
        self.updates_done += 1

        return loss_dict    

    # generate best action randomly
    def is_best_action_random_generator(self, batch_size):
        p = 1.0 / float(self.action_number)
        
        # [0,1)
        prob_array = torch.rand(batch_size, device=self.device)
        is_best_action_array = (prob_array <= p)

        return is_best_action_array

    # train for one update (our algorithm)
    def train_one_update_ours(self):
        assert self.reward_type == "minus_one_zero", "Error: our algorithm assumes reward type to be minus one zero"
        
        # switch model mode
        self.q_network.train()
        
        # (next)observations # (B,C,H,W)
        # actions # (B)
        # rewards # (B)
        # goals # (B,goal_dim)
        # dones # (B)
        observations, goals, actions, rewards, next_observations, next_goals, dones, next_actions, optimal_actions = self.train_dataset.get_transition_batch(self.batch_size)


        # compute target Q
        with torch.no_grad():
            batch_size = observations.size()[0]
            non_optimal_actions = ~optimal_actions
            non_optimal_action_num = torch.count_nonzero(non_optimal_actions).item()


            Q_targets_next = self.q_network(next_observations, next_goals).detach()
            Q_target_best_next = torch.gather(Q_targets_next,
                                dim=1,
                                index=next_actions.long().unsqueeze(1)).squeeze(1) # [B]

            Q_targets = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            # target Q: optimal action
            Q_targets[optimal_actions] = rewards[optimal_actions] + self.gamma * Q_target_best_next[optimal_actions] * (1.0 - dones.float()[optimal_actions]) 
            
            # target Q: non optimal action
            non_optimal_rewards = (1+self.gamma)*(-1.0 * self.negative_reward_scale * torch.ones(non_optimal_action_num, device=self.device)) + pow(self.gamma, 2) * rewards[non_optimal_actions]
            Q_targets[non_optimal_actions] = non_optimal_rewards + pow(self.gamma, 3) * Q_target_best_next[non_optimal_actions] * (1.0 - dones.float()[non_optimal_actions])
                
            Q_targets = Q_targets.detach() #[B]
            
        # compute predicted Q
        # [B,1] -> [B]
        Q_output = self.q_network(observations, goals)
        Q_predicted = torch.gather(Q_output,
                                    dim=1,
                                    index=actions.long().unsqueeze(1)).squeeze(1) 

        # compute Q loss
        q_loss = F.mse_loss(Q_predicted, Q_targets) # a single float number

        # optimize Q network
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()
        
        # record losses
        loss_dict = {}
        loss_dict["q_loss"] = q_loss.detach().cpu().item()
        
        # count how many actions are predicted correctly
        with torch.no_grad():
            predicted_actions = torch.argmax(Q_output.detach(), dim=1)
            # print(predicted_actions)
            # print("="*30)
            compare_actions = torch.eq(predicted_actions, actions)
            # print(actions)
            # print("="*30)
            # print(compare_actions)
            # print("="*30)
            action_pred_correct = torch.sum(compare_actions.int())
            loss_dict["correct_action_num"] = action_pred_correct.detach().cpu().item()
            # print(loss_dict["correct_action_num"])
            # print("="*30)
            # exit()
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
        #print("goal form ==========> %s"%(self.config.get("goal_form")))

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
        train_action_correct_nums = []

        if self.with_bc_loss:
            train_total_losses = []
            train_bc_losses = []
        
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
            if self.q_learning_type == "ours":
                loss_dict = self.train_one_update_ours()
            else:    
                loss_dict = self.train_one_update_others()

            # record losses
            train_q_losses.append(loss_dict["q_loss"]) 
            train_action_correct_nums.append(loss_dict["correct_action_num"])
            if self.with_bc_loss:
                train_bc_losses.append(loss_dict["bc_loss"])
                train_total_losses.append(loss_dict["loss"])
                

        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.start_time

        logs['training/q_loss_mean'] = np.mean(train_q_losses)
        logs['training/q_loss_std'] = np.std(train_q_losses)

        logs['training/action_accuracy'] = np.sum(train_action_correct_nums) / float(self.train_dataset.total_transition_num())


        if self.with_bc_loss:
            logs['training/bc_loss_mean'] = np.mean(train_bc_losses)
            logs['training/bc_loss_std'] = np.std(train_bc_losses)
            logs['training/total_loss_mean'] = np.mean(train_total_losses)
            logs['training/total_loss_std'] = np.std(train_total_losses)

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
    trainer = DQNTrainer(
        config_filename="imitation_learning_dqn.yaml", 
        resume=True,
        resume_experiment_name="s1-20221005-174833",
        resume_ckpt_index=10)
    trainer.train()
