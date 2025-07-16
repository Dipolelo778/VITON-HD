# network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------
# 1️⃣ Geometric Matching Module (GMM)
# -------------------------------------------
class GMM(nn.Module):
    def __init__(self):
        super(GMM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(22, 64, 7, padding=3),  # 18 pose + 3 cloth mask + 1 agnostic mask
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU()
        )
        self.flow = nn.Conv2d(64, 2, 3, padding=1)  # Output flow field (dx, dy)

    def forward(self, agnostic, cloth_mask, pose_map):
        x = torch.cat([agnostic, cloth_mask, pose_map], dim=1)
        feat = self.conv(x)
        flow = self.flow(feat)
        return flow

# -------------------------------------------
# 2️⃣ Try-On Refiner (U-Net)
# -------------------------------------------
class TryOnUNet(nn.Module):
    def __init__(self):
        super(TryOnUNet, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1), nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU()
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1), nn.Tanh()
        )

    def forward(self, warped_cloth, person_image):
        x = torch.cat([warped_cloth, person_image], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        d3 = self.dec3(e3)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        out = self.dec1(torch.cat([d2, e1], dim=1))
        return out

# -------------------------------------------
# 3️⃣ Optional Discriminator for GAN
# -------------------------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 0)
        )

    def forward(self, img):
        return self.model(img)

# -------------------------------------------
# ✅ Done: GMM + Refiner + (Optional) Discriminator
# -------------------------------------------
                      


