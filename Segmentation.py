# segmentation.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class SimpleSegmentationNet(nn.Module):
    """
    For demo: A placeholder. 
    In production, swap with a real pre-trained human parsing network 
    like SCHP or LIP.
    """
    def __init__(self, num_classes=20):
        super(SimpleSegmentationNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        return self.net(x)

class Segmenter:
    def __init__(self, model_path=None):
        self.model = SimpleSegmentationNet().eval()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.transform = transforms.Compose([
            transforms.Resize((256, 192)),
            transforms.ToTensor()
        ])

    def segment(self, image_path):
        img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
            segmentation_map = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        return segmentation_map  # shape: (H, W), values = class IDs

if __name__ == "__main__":
    seg = Segmenter()
    seg_map = seg.segment("./sample_person.jpg")
    print(np.unique(seg_map))  # Shows which parts/classes were found
