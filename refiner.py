# refiner.py

import torch
import torch.nn as nn

class RefinerUNet(nn.Module):
    def __init__(self):
        super(RefinerUNet, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        return self.dec(self.enc(x))


