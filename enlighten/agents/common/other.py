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

# compute discounted cumulative future reward for each step in reward list x
def discount_cumsum(self, x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    # from the last to the first DP
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum