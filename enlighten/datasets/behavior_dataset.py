import random
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

class BehaviorDataset(TorchDataset):
    """ Sample behavior segments for supervised learning 
    from given input episodes.
    """
    def __init__(self, config, episodes, teacher):
        super(BehaviorDataset, self).__init__()
        self.episodes = episodes
        self.config = config
        # the number of data points in this dataset is not the size of the dataset
        # the size of the dataset = number of batches * batch_size
        # each data is a (s,a) tuple sampled from one episode
        # the size of dataset is not the number of episodes
        self.size = int(config.get("batch_size")) * int(config.get("num_updates_per_iter"))
        self.teacher = teacher

    def __len__(self):
        return self.size

    def get_target(self):
        pass    

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