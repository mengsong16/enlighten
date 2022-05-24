import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys
import os

from enlighten.agents.models.decision_transformer import DecisionTransformer
from enlighten.agents.trainer.seq_trainer import SequenceTrainer
from enlighten.agents.evaluation.evaluate_episodes import eval_episodes
from enlighten.utils.path import *



# variant: config of wandb
def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    # group_name - 6 digit random number
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
    elif env_name == 'reacher2d':
        from eva.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    else:
        raise NotImplementedError

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load all trajectories from a specific dataset
    dataset_path = os.path.join(data_path, f'{env_name}-{dataset}-v2.pkl')
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # parse all path information into separate lists of 
    # states (observations), traj_lens, returns
    mode = variant.get('mode', 'normal')
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

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.) # percentage of top trajectories

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
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4*variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )
    

    model = model.to(device=device)

    # create optimizer: AdamW
    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
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
    
    # logging
    project_name = 'decision-transformer'
    if log_to_wandb:
        # initialize this run under project xxx
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project=project_name,
            config=variant,
            dir=os.path.join(root_path)
        )

        #print(exp_prefix)
        #exit()
        # wandb.watch(model)  # wandb has some bug

    # train for max_iters iterations
    # each iteration includes num_steps_per_iter steps
    for iter in range(variant['max_iters']):
        outputs = trainer.train_one_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
