import os
import torch
import pickle
import numpy as np
from pathlib import Path

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)

def make_parent(file_path):
    file_path = Path(file_path)
    os.makedirs(file_path.parent, exist_ok=True)

def pickle_dump(python_object, file_path):
    make_parent(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(python_object, f)

def pickle_load(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
