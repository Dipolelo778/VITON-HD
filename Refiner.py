# refiner.py

import torch.nn as nn

class Refiner(nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)
