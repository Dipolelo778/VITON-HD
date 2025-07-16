# gmm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GMM(nn.Module):
    def __init__(self):
        super(GMM, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 64, 4, 2, 1),  # person + pose + cloth mask
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 2, 4, 2, 1)  # flow field (dx, dy)
        )

    def forward(self, person, pose, cloth_mask, cloth):
        x = torch.cat([person, pose, cloth_mask], dim=1)  # 3+2+2 channels
        flow_field = self.decoder(self.encoder(x))  # Bx2xHxW

        # warp the cloth with the predicted flow field
        B, C, H, W = cloth.size()
        grid_X, grid_Y = torch.meshgrid(
            torch.linspace(-1, 1, W),
            torch.linspace(-1, 1, H)
        )
        grid = torch.stack((grid_X, grid_Y), 2).permute(1, 0, 2).unsqueeze(0).to(cloth.device)
        grid = grid + flow_field.permute(0, 2, 3, 1)

        warped_cloth = F.grid_sample(cloth, grid, padding_mode='border', align_corners=True)
        return warped_cloth
