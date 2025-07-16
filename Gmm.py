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
            nn.ConvTranspose2d(64, 2, 4, 2, 1)  # output flow field (dx, dy)
        )

    def forward(self, person, pose, cloth_mask, cloth):
        # Concatenate all inputs along channels
        x = torch.cat([person, pose, cloth_mask], dim=1)
        flow_field = self.decoder(self.encoder(x))  # B x 2 x H x W

        # Create sampling grid
        B, C, H, W = cloth.size()
        grid_X, grid_Y = torch.meshgrid(
            torch.linspace(-1, 1, W, device=cloth.device),
            torch.linspace(-1, 1, H, device=cloth.device)
        )
        grid = torch.stack((grid_X, grid_Y), 2).permute(1, 0, 2).unsqueeze(0)
        grid = grid + flow_field.permute(0, 2, 3, 1)

        warped_cloth = F.grid_sample(cloth, grid, padding_mode='border', align_corners=True)
        return warped_cloth, flow_field
