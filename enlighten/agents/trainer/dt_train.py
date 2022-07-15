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
        observations, actions, goals, timesteps, attention_mask = self.train_dataset.get_batch(self.batch_size)
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


if __name__ == '__main__':
    trainer = DTTrainer(config_filename="imitation_learning_dt.yaml")
    trainer.train()
