# config.py

import os
import torch

class Config:
    IMG_HEIGHT = 256
    IMG_WIDTH = 192
    BATCH_SIZE = 4
    EPOCHS = 50
    LR = 2e-4
    DEVICE = "cuda"
    DATA_ROOT = "./data"
    CHECKPOINTS = "./checkpoints"
    os.makedirs(CHECKPOINTS, exist_ok=True)
