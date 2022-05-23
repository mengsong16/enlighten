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

# compute discounted cumulative future reward for each step in reward list x
def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    # from the last to the first DP
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

# sample a batch
def get_batch(batch_size=256, max_len=K):
    # sample batch_size trajectories from the trajectory pool with replacement
    # prefer long trajectory
    batch_inds = np.random.choice(
        np.arange(num_trajectories),
        size=batch_size,
        replace=True,
        p=p_sample,  # reweights so we sample according to timesteps
    )

    # separate a trajectory batch into state, action, reward, discounted return, timestep, mask batch
    # each element in the new batch is a trjectory segment, max_len: segment length which will be used to train sequence model
    s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
    for i in range(batch_size):
        # current trajectory
        traj = trajectories[int(sorted_inds[batch_inds[i]])]
        # randomly pick a segment of length max_len from current trajectory starting from state si
        si = random.randint(0, traj['rewards'].shape[0] - 1)

        # Note that if si+max_len exceed current traj length, only get elements until the episode ends
        s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
        a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
        r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
        # d: dones (true or false)
        if 'terminals' in traj:
            d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
        else:
            d.append(traj['dones'][si:si + max_len].reshape(1, -1))
        # each timestep is the step index inside this segment: e.g. [5,6,7]
        # s[-1].shape[1] <= max_len
        timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        # if actual index exceed predefined max episode length, use the last step index (i.e. index max_ep_len - 1) instead
        # timesteps[-1]: current segment
        # timesteps[-1] >= max_ep_len: for each step in current segment, check whether it exceeds max_ep_len
        timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff index
        # undiscounted return since gamma = 1
        # first compute for each state until the episode ends, then cut off for the current segment
        rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
        # pad with a single 0 reward for the last state
        if rtg[-1].shape[1] <= s[-1].shape[1]: # always true??
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

        # pre-padding and state + reward normalization
        # tlen is the true length of current segment
        tlen = s[-1].shape[1]
        # pad state with 0 if shorter than max_len
        s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
        # normalize state distribution to N(0,1)
        s[-1] = (s[-1] - state_mean) / state_std
        # pad action with -10 if shorter than max_len
        a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
        # pad reward with 0 if shorter than max_len
        r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
        # pad dones with 2 if shorter than max_len
        d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
        # pad rtg with 0 if shorter than max_len
        # scale rtg by scale
        rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
        # pad timestep with 0 if shorter than max_len
        timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
        # mask = 1 (not done) until tlen, after that = 0 (done)
        mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

    # numpy to torch tensor
    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
    d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

    return s, a, r, d, rtg, timesteps, mask

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

    if model_type == 'bc':
        # only keep the first target
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

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
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2), # only use l2 loss on actions
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
