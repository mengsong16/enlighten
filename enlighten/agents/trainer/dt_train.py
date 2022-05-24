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

from enlighten.agents.models.decision_transformer import DecisionTransformer
from enlighten.agents.trainer.seq_trainer import SequenceTrainer
from enlighten.agents.evaluation.evaluate_episodes import eval_episodes
from enlighten.utils.path import *
from enlighten.utils.config_utils import parse_config
from enlighten.agents.common.seed import set_seed_except_env_seed
from enlighten.agents.common.other import get_device

class DTTrainer:
    def __init__(self, config_filename="imitation_learning.yaml"):
        assert config_filename is not None, "needs config file to initialize trainer"
        
        config_file = os.path.join(config_path, config_filename)
        self.config = parse_config(config_file)

        # seed everything except env
        self.seed = int(self.config.get("seed"))
        set_seed_except_env_seed(self.seed)

        #self.env = create_env(self.config.get("env_id")) 

        # set device
        self.device = get_device(self.config)
    
        # set experiment name
        self.set_experiment_name()

        # init wandb
        self.log_to_wandb = self.config.get("log_to_wandb")
        if self.log_to_wandb:
            self.init_wandb()
    
    def set_experiment_name(self):
        self.project_name = 'dt'.lower()

        self.group_name = self.config.get("experiment_name").lower()

        # experiment_name: seed-YearMonthDay-HourMiniteSecond
        now = datetime.datetime.now()
        self.experiment_name = "s%d-"%(self.seed)+now.strftime("%Y%m%d-%H%M%S").lower() 

    def init_wandb(self):
        # initialize this run under project xxx
        wandb.init(
            name=self.experiment_name,
            group=self.group_name,
            project=self.project_name,
            config=self.config,
            dir=os.path.join(root_path)
        )

    # self.config.get: config of wandb
    def train(self):
        # load all trajectories from a specific dataset
        dataset_path = os.path.join(data_path, f'{env_name}-{dataset}-v2.pkl')
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

        # parse all path information into separate lists of 
        # states (observations), traj_lens, returns
        states, traj_lens, returns = [], [], []
        for path in trajectories:
            # the last step: return R, previous steps: 0
            if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
                path['rewards'][-1] = path['rewards'].sum()
                path['rewards'][:-1] = 0.
            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            # return is not discounted
            returns.append(path['rewards'].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # compute mean and standard deviation over states from all trajectories
        # used for input normalization
        # avoid 0 by adding 1e-6
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # total number of steps of all trajectories
        num_timesteps = sum(traj_lens)

        # print basic info of experiment run
        print('=' * 50)
        print(f'Starting new experiment: {env_name} {dataset}')
        print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
        print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
        print('=' * 50)

        K = int(self.config.get('K'))
        batch_size = int(self.config.get('batch_size'))
        num_eval_episodes = int(self.config.get('num_eval_episodes'))
        pct_traj = float(self.config.get.get('pct_traj')) # percentage of top trajectories

        # only train on top pct_traj trajectories (for BC experiment)
        num_timesteps = max(int(pct_traj*num_timesteps), 1)
        # sort trajectories from lowest to highest return
        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        # get the number of total timesteps of top pct_traj trajectories
        timesteps = traj_lens[sorted_inds[-1]]
        # ind iterate from the last to the first
        ind = len(trajectories) - 2
        # the total steps should not exceed num_timesteps
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        # only keep the top percentage trajectory indices    
        sorted_inds = sorted_inds[-num_trajectories:]

        # used to reweight sampling so we sample according to timesteps instead of trajectories
        # p_sample is a list of step percentage for each trajectory
        p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

        # create model
        model = DecisionTransformer(
            state_dim=env.observation_space,
            act_dim=int(self.config.get("action_number")),
            max_length=K,
            max_ep_len=int(self.config.get("max_ep_len")),  
            hidden_size=int(self.config.get('embed_dim')), # parameters starting from here will be passed to gpt2
            n_layer=int(self.config.get('n_layer')),
            n_head=int(self.config.get('n_head')),
            n_inner=int(4*self.config.get('embed_dim')),
            activation_function=self.config.get('activation_function'),
            n_positions=1024,
            resid_pdrop=float(self.config.get('dropout')),
            attn_pdrop=float(self.config.get('dropout')),
        )
        

        model = model.to(device=self.device)

        # create optimizer: AdamW
        warmup_steps = int(self.config.get('warmup_steps'))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config.get('learning_rate')),
            weight_decay=float(self.config.get('weight_decay')),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)
        )
        
        # create trainer
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            scheduler=scheduler,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

        # train for max_iters iterations
        # each iteration includes num_steps_per_iter steps
        for iter in range(int(self.config.get('max_iters'))):
            outputs = trainer.train_one_iteration(num_steps=int(self.config.get('num_steps_per_iter')), iter_num=iter+1, print_logs=True)
            if self.log_to_wandb:
                wandb.log(outputs)


if __name__ == '__main__':
    trainer = DTTrainer(config_filename="imitation_learning.yaml")
    trainer.train()
