import torch
import random
import numpy as np

def set_seed(seed, env):
    #seed %= 4294967294
    
    set_seed_except_env_seed(seed)

    env.seed(seed)
    env.action_space.seed(seed)

def set_seed_except_env_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
