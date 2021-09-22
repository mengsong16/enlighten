import torch
import random
import numpy as np

def set_seed(seed, env):
    seed %= 4294967294
    
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

    env.seed(seed)
    env.action_space.seed(seed)