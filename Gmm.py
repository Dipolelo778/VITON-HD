# gmm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TPSGridGen(nn.Module):
    """
    Thin-Plate Spline grid generator.
    Warps a grid using learned TPS parameters.
    """
    def __init__(self, target_height, target_width, grid_size=3):
        super(TPSGridGen, self).__init__()
        self.height = target_height
        self.width = target_width
        self.grid_size = grid_size

        # Control points (grid_size x grid_size)
        self.num_ctrl_pts = grid_size * grid_size
        self.target_control_points = self._build_control_points()

    def _build_control_points(self):
        """Create normalized grid points."""
        axis_coords = torch.linspace(-1, 1, self.grid_size)
        p = torch.stack(torch.meshgrid([axis_coords, axis_coords]), dim=-1)
        return p.view(-1, 2)

    def forward(self, theta):
        """
        theta: [batch_size, num_ctrl_pts * 2]
        """
        # Use grid_sample or custom TPS solver — here just a placeholder identity grid
        batch_size = theta.size(0)
        grid = F.affine_grid(torch.eye(2,3).unsqueeze(0).expand(batch_size,-1,-1),
                             torch.Size((batch_size, 3, self.height, self.width)),
                             align_corners=True)
        return grid

class GMM(nn.Module):
    def __init__(self, grid_size=3):
        super(GMM, self).__init__()
        self.extraction = nn.Sequential(
            nn.Conv2d(22, 64, 7, padding=3), nn.ReLU(),
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(),
            nn.Conv2d(128, 256, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, grid_size * grid_size * 2)
        self.tps = TPSGridGen(256, 192, grid_size)

    def forward(self, agnostic, cloth):
        """
        agnostic: person image without the target clothing region (pose + mask)
        cloth: target clothing image
        """
        input = torch.cat([agnostic, cloth], dim=1)
        features = self.extraction(input)
        features = features.view(features.size(0), -1)
        theta = self.fc(features)
        grid = self.tps(theta)
        warped_cloth = F.grid_sample(cloth, grid, align_corners=True)
        return warped_cloth, grid

if __name__ == "__main__":
    gmm = GMM()
    agnostic = torch.randn(1, 20, 256, 192)  # pose + mask
    cloth = torch.randn(1, 3, 256, 192)
    warped, grid = gmm(agnostic, cloth)
    print(warped.shape)
