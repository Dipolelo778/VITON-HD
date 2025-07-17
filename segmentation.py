# segmentation.py

import torch
import torch.nn as nn

class SegmentationNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SegmentationNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
