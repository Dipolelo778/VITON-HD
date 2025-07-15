# config.py

import os

class Config:
    data_root = "./data"
    epochs = 50
    batch_size = 4
    lr = 2e-4
    device = "cuda"
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
