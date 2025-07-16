# cloth_parser.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ClothParser(nn.Module):
    def __init__(self):
        super(ClothParser, self).__init__()
        # Use pretrained DeepLabV3 for semantic segmentation
        self.backbone = models.segmentation.deeplabv3_resnet50(pretrained=True).eval()

    def forward(self, image_tensor):
        """
        Args:
            image_tensor: [B, 3, H, W] input cloth image
        Returns:
            mask: [B, 1, H, W] binary mask for cloth
        """
        output = self.backbone(image_tensor)["out"]
        mask = torch.argmax(output, dim=1, keepdim=True).float()
        return mask

def parse_cloth(image_path):
    transform = transforms.Compose([
