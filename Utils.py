# utils.py

import torch

def to_device(data, device):
    return [d.to(device) for d in data]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
