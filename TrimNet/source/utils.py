import os
import random

import numpy as np
import torch
from torch import nn



class Option(object):
    def __init__(self, d):
        self.__dict__ = d


def save_print_log(msg, save_dir=None, show=True):
    with open(save_dir + "/log.txt", "a+") as f:
        f.write(msg + "\n")
        if show:
            print(msg)


def seed_set(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".
                        format(x, allowable_set))
    return [x == s for s in allowable_set]


def onehot_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]
