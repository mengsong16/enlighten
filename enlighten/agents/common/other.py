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