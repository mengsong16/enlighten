import os
import numpy as np
import yaml
import math
import collections
import torch

def get_device(config):
    if torch.cuda.is_available():
        return torch.device("cuda:{}".format(int(config.get("gpu_id"))))
    else:
        return torch.device("cpu")

# # compute discounted cumulative future reward for each step in reward list x
# def discount_cumsum(self, x, gamma):
#     discount_cumsum = np.zeros_like(x)
#     discount_cumsum[-1] = x[-1]
#     # from the last to the first DP
#     for t in reversed(range(x.shape[0]-1)):
#         discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
#     return discount_cumsum

def get_obs_channel_num(config):
    obs_channel = 0
    if config.get("color_sensor"):
        obs_channel += 3
    if config.get("depth_sensor"):
        obs_channel += 1
    return obs_channel

# given reward list reward_list
# compute discounted cumulative future reward for each step in reward list x
def discount_cumsum(reward_list, gamma):
    discount_cumsum = np.zeros_like(reward_list)
    # step n
    discount_cumsum[-1] = reward_list[-1]
    # from step n-1 to the first step DP
    for t in reversed(range(reward_list.shape[0]-1)):
        discount_cumsum[t] = reward_list[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

# action_seq_length: the number of actions on a optimal trajectory starting from the current state
# assume that the trajectory leads to success
def get_reward_after_action(action_seq_length, positive_reward, negative_reward_scale):
    assert negative_reward_scale > 0
    assert positive_reward >= 0

    reward_list = np.ones((action_seq_length,), dtype="float32") * negative_reward_scale * (-1)
    reward_list[-1] = positive_reward

    return reward_list

# action_seq_length: the number of actions on a optimal trajectory starting from the current state
def get_optimal_q(action_seq_length, gamma, positive_reward, negative_reward_scale):
    reward_list = get_reward_after_action(action_seq_length, positive_reward, negative_reward_scale)
    q_list = discount_cumsum(reward_list, gamma)

    return q_list[0]

def get_geodesic_distance_based_q(env):
    # geodesic distance from current state to goal state
    d = env.get_current_distance()
    q = -d
    return q