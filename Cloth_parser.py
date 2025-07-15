# cloth_parser.py

import torch.nn as nn

class ClothParser(nn.Module):
    def __init__(self):
        super(ClothParser, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, cloth):
        return self.net(cloth)
