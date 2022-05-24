import numpy as np
import torch
import time
import random
from torch.nn import functional as F

# train seq2seq imitation learning
class SequenceTrainer():
    def __init__(self, model, optimizer, batch_size, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()

    # train for one iteration
    def train_one_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        # switch model to training mode
        self.model.train()

        # train for num_steps
        for _ in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        # switch model to evaluation mode
        self.model.eval()
        # evaluate by each evaluation function
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    # train for one step
    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        action_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        # loss is evaluated only on actions
        loss =  F.cross_entropy(action_preds, action_target)

        self.optimizer.zero_grad()
        # compute weight gradients
        loss.backward()
        # clip weight grad
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        # optimize for one step
        self.optimizer.step()

        # compute action prediction error
        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
    
    # compute discounted cumulative future reward for each step in reward list x
    def discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        # from the last to the first DP
        for t in reversed(range(x.shape[0]-1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
        return discount_cumsum

    # sample a batch
    def get_batch(self, batch_size=256, max_len=K):
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
            rtg.append(self.discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
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
