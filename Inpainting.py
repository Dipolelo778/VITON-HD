# inpainting.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class InpaintingUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(InpaintingUNet, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )

        self.middle = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for _ in range(4)]
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1), nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        m = self.middle(e3)

        d3 = self.dec3(m)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)

        out = self.dec1(d2)
        return out

if __name__ == "__main__":
    # Example: image + mask
    x = torch.randn(1, 4, 256, 192)  # 3 channels image + 1 mask
    model = InpaintingUNet()
    y = model(x)
    print(y.shape)
