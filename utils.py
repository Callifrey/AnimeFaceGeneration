import torch
import numpy as np

def smooth_pos_label(y):
    t = np.random.random(y.shape)
    y = y - 0.3 + t * 0.5
    y = torch.tensor(y, dtype=torch.float)
    return y

def smooth_neg_label(y):
    t = np.random.random(y.shape)
    y = y + t *0.3
    y = torch.tensor(y, dtype=torch.float)
    return y