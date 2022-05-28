import random
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from enlighten.agents.common.other import discount_cumsum, get_obs_channel_num
import pickle

class BehaviorDataset:
    """ Sample trajectory segments for supervised learning 
    """
    def __init__(self, config, device):
        self.config = config  # config is a dictionary
        self.load_trajectories()
        self.device = device
        self.context_length = int(self.config.get("K"))
        self.max_ep_len = int(self.config.get("max_ep_len")) 
        self.goal_dim = int(self.config.get("goal_dimension")) 
        self.obs_channel = get_obs_channel_num(self.config)
        if self.obs_channel == 0:
            print("Error: channel of observation input to the encoder is 0")
            exit()
        self.obs_width = int(self.config.get("image_width")) 
        self.obs_height = int(self.config.get("image_height"))

    def load_trajectories(self):
        # load all trajectories from a specific dataset
        dataset_path = self.config.get("behavior_dataset_path")
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        self.num_trajectories = len(self.trajectories)

        print("Loaded %d trajectories"%(self.num_trajectories))
        
        
    # sample a batch
    def get_batch(self, batch_size=256):
        # sample batch_size trajectories from the trajectory pool with no replacement
        batch_inds = np.random.choice(
            np.arange(self.num_trajectories),
            size=batch_size,
            replace=False
        )
        # organize a batch into observation, action, goal, distance to goal, timestep, mask
        # each element in the new batch is a trjectory segment, max_len: segment length which will be used to train sequence model
        o, a, g, d, dtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            # current trajectory
            traj = self.trajectories[int(batch_inds[i])]
            # randomly pick a segment of context length from current trajectory starting from index si
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # Note that if si+self.context_length exceed current traj length, only get elements until the episode ends
            o.append(traj['observations'][si:si + self.context_length].reshape(1, -1, self.obs_obs_channel, self.obs_height, self.obs_width))
            a.append(traj['actions'][si:si + self.context_length].reshape(1, -1))
            g.append(traj['goals'][si:si + self.context_length].reshape(1, -1, self.goal_dim))
            d.append(traj['dones'][si:si + self.context_length].reshape(1, -1))
            dtg.append(traj['distance_to_goals'][si:si + self.context_length].reshape(1, -1))

            # tlen is the true length of current segment
            # tlen <= self.context_length
            tlen = o[-1].shape[1]

            # each timestep is the step index inside this segment: e.g. [5,6,7]
            timesteps.append(np.arange(si, si + tlen).reshape(1, -1))
            # if actual index exceeds predefined max episode length, use the last step index (i.e. index max_ep_len - 1) instead
            # if timesteps in current segment >= self.max_ep_len: for each step in current segment, check whether it exceeds self.max_ep_len
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len-1  
            
            # pad with a single 0 reward for the last state??
            if dtg[-1].shape[1] <= tlen: # always true??
                dtg[-1] = np.concatenate([dtg[-1], np.zeros((1, 1, 1))], axis=1)

            # mask = 1 (attend to not paddding part) until tlen
            mask.append(np.ones((1, tlen)))

            # left padding current segment to self.context_length if shorter than self.context_length
            op, ap, gp, dp, dtgp, tp, mp = self.get_padding(self.context_length - tlen)
             
            o[-1] = np.concatenate([op, o[-1]], axis=1)
            a[-1] = np.concatenate([ap, a[-1]], axis=1)
            g[-1] = np.concatenate([gp, g[-1]], axis=1)
            d[-1] = np.concatenate([dp, d[-1]], axis=1)
            dtg[-1] = np.concatenate([dtgp, dtg[-1]], axis=1)
            timesteps[-1] = np.concatenate([tp, timesteps[-1]], axis=1)
            mask[-1] = np.concatenate([mp, mask[-1]], axis=1)

        # numpy to torch tensor
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=self.device)
        g = torch.from_numpy(np.concatenate(g, axis=0)).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=self.device)
        dtg = torch.from_numpy(np.concatenate(dtg, axis=0)).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)

        return o, a, g, d, dtg, timesteps, mask

    # get padding as numpy array
    def get_padding(self, padding_length):
        # pad observation with 0
        op = np.zeros((1, padding_length, self.obs_channel, self.obs_height, self.obs_width))
        # pad action with 0 (stop)
        ap = np.ones((1, padding_length))
        # pad goal with 0 
        gp = np.zeros((1, padding_length, self.goal_dim))
        # pad dones with 2
        dp = np.ones((1, padding_length)) * 2
        # pad dtg with 0
        dtgp = np.zeros((1, padding_length, 1))
        # pad timestep with 0
        tp = np.zeros((1, padding_length))
        # pad mask with 0 (not attend to)
        mp = np.zeros((1, padding_length))

        return op, ap, gp, dp, dtgp, tp, mp
