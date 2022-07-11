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

from enlighten.agents.models.rnn_seq_model import RNNSequenceModel
from enlighten.agents.trainer.seq_trainer import SequenceTrainer
from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.agents.common.seed import set_seed_except_env_seed
from enlighten.agents.common.other import get_device
from enlighten.datasets.behavior_dataset import BehaviorDataset
from enlighten.envs.multi_nav_env import MultiNavEnv
from enlighten.agents.common.other import get_obs_channel_num
from enlighten.agents.evaluation.evaluate_episodes import MultiEnvEvaluator

class RNNTrainer(SequenceTrainer):
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
            rnn_type=self.config.get('rnn_type')
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
       
        observations, action_targets, prev_actions, goals, batch_sizes = self.train_dataset.get_batch(self.batch_size)
        
        # print(observations.size())
        # print(action_targets.size())
        # print(prev_actions.size())
        # print(goals.size())
        # print(batch_sizes.size())

        # create h0: [1, B, hidden_size]
        rnn_hidden_size = int(self.config.get('rnn_hidden_size'))
        h_0 = torch.zeros(1, self.batch_size, rnn_hidden_size, dtype=torch.float32, device=self.device) 


        # action_preds: [T, action_num]
        action_preds = self.model.forward(observations, prev_actions, goals, h_0, batch_sizes)

        # loss is computed over the whole sequence (K action tokens)
        # action_target are ground truth action indices (not one-hot vectors)
        loss =  F.cross_entropy(action_preds, action_targets)

        #print(loss) # a single float number

        self.optimizer.zero_grad()
        # compute weight gradients
        loss.backward()
        
        # optimize for one step
        self.optimizer.step()

        return loss.detach().cpu().item()

if __name__ == '__main__':
    trainer = RNNTrainer(config_filename="imitation_learning_rnn.yaml")
    trainer.train()
