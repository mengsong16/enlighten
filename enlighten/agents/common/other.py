import os
import numpy as np
import yaml
import math
import collections
from collections import OrderedDict
import torch
from gym.spaces import Box, Discrete, Tuple

class Number(metaclass=ABCMeta):
    """All numbers inherit from this class.

    If you just want to check if an argument x is a number, without
    caring what kind, use isinstance(x, Number).
    """
    __slots__ = ()

    # Concrete numeric types must provide their own hash implementation
    __hash__ = None
    
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

def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))

def add_prefix(log_dict: OrderedDict, prefix: str, divider=''):
    with_prefix = OrderedDict()
    for key, val in log_dict.items():
        with_prefix[prefix + divider + key] = val
    return with_prefix

def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v

# polyak update
# tau = 1: 100% copy from source to target
def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def from_numpy(device, *args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return from_numpy(elem_or_tuple).float()

def np_to_pytorch_batch(np_batch):
    if isinstance(np_batch, dict):
        return {
            k: _elem_or_tuple_to_variable(x)
            for k, x in _filter_batch(np_batch)
            if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
        }
    else:
        _elem_or_tuple_to_variable(np_batch)

def zeros(*sizes, torch_device=None, **kwargs):
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def ones(*sizes, torch_device=None, **kwargs):
    return torch.ones(*sizes, **kwargs, device=torch_device)

def randint(*sizes, torch_device=None, **kwargs):
    return torch.randint(*sizes, **kwargs, device=torch_device)


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{}{}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats