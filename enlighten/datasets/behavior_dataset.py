import random
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from enlighten.agents.common.other import discount_cumsum

class BehaviorDataset(TorchDataset):
    """ Sample trajectory segments for supervised learning 
    """
    def __init__(self, config, episodes):
        super(BehaviorDataset, self).__init__()
        self.episodes = episodes
        self.config = config
        # the number of data points in this dataset is not the size of the dataset
        # the size of the dataset = number of batches * batch_size
        # each data is a (s,a) tuple sampled from one episode
        # the size of dataset is not the number of episodes
        self.size = int(config.get("batch_size")) * int(config.get("num_updates_per_iter"))

    def __len__(self):
        return self.size

    def load_trajectories(self):
        # load all trajectories from a specific dataset
        dataset_path = os.path.join(data_path, f'{env_name}-{dataset}-v2.pkl')
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

        # parse loaded trajectories into separate lists
        states, traj_lens, returns = self.parse_trajectories(trajectories)

        # total number of steps of all trajectories
        num_timesteps = sum(traj_lens)
        
    def pad_segment(self, traj): 
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
   
    # sample a batch
    def get_batch(self, trajectories, max_ep_len, batch_size=256, max_len=K):
        # sample batch_size trajectories from the trajectory pool with replacement
        # prefer long trajectory
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            #p=p_sample,  # reweights so we sample according to timesteps
        )

        # separate a trajectory batch into state, action, reward, discounted return, timestep, mask batch
        # each element in the new batch is a trjectory segment, max_len: segment length which will be used to train sequence model
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            # current trajectory
            #traj = trajectories[int(sorted_inds[batch_inds[i]])]
            traj = trajectories[int(batch_inds[i])]
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

    # parse all path information into separate lists of 
    # states (observations), traj_lens, returns
    def parse_trajectories(self, trajectories):
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
        states = np.concatenate(states, axis=0)

        return states, traj_lens, returns   

    # each batch is sampled independently, so different batches can include the same data point
    def __getitem__(self, idx):
        # get episode
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]

        # randomly sample an episode
        episode = random.choice(self.episodes)
        S, A, R, S_ = episode

        # randomly select a state
        episode_len = S.shape[0]
        start_index = np.random.choice(episode_len - 1) # ensures cmd_steps >= 1
        
        # get achieved target
        target = self.teacher.get_achieved_target(episode_len, start_index, R)

        # construct a sample: (aug_state, gt_action)
        aug_state = augment_state(S[start_index,:], target)

        if aug_state.ndim > 1:
            aug_state = np.squeeze(aug_state, axis=0)
        
        # ground truth action
        if A.ndim == 1:  # discrete action
            gt_action = A[start_index]
        else:  # continuous action
            gt_action = A[start_index,:]
            if gt_action.ndim > 1:
                gt_action = np.squeeze(gt_action, axis=0)

        # convert to torch tensor
        # aug_state: (state_dim,)
        # gt_action: (action_dim,)
        # After stacking (stack will create an extra dimension):
        # aug_state: (B, state_dim)
        # gt_action: (B, action_dim)
        sample = {
            'augmented_state': torch.tensor(aug_state, dtype=torch.float), 
            'ground_truth_action': torch.tensor(gt_action, dtype=torch.float) 
        }        
        return sample