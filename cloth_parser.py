import torch
import torch.nn as nn

class ClothParser(nn.Module):
    def __init__(self):
        super(ClothParser, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
