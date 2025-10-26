# This file contains additional utility classes and functions
import os
import torch
import random
import numpy as np


# Hyperparameter container
class Hyperparameter:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Sets seed to ensure reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
