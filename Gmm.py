# gmm.py

import torch
import torch.nn as nn

class GMM(nn.Module):
    def __init__(self):
        super(GMM, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(6, 64, 7, 1, 3), nn.ReLU(),
            nn.Conv2d(64, 128, 5, 1, 2), nn.ReLU(),
            nn.Conv2d(128, 2, 3, 1, 1)  # output TPS grid params
        )

    def forward(self, person, cloth):
        x = torch.cat([person, cloth], dim=1)
        grid_params = self.main(x)
        # Fake: you’d use grid_sample and TPS warp in real.
        return grid_params
