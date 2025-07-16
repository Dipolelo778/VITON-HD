import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import cv2

# -----------------------------
# Transform: PIL <-> Tensor
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((256, 192)),
    transforms.ToTensor(),
])

to_pil = transforms.ToPILImage()

# -----------------------------
# Load image
# -----------------------------
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    return transform(img)  # Tensor [C,H,W]

# -----------------------------
# Save tensor to image
# -----------------------------
def save_image(tensor, path):
    img = tensor.detach().cpu().clamp(0, 1)
    img_pil = to_pil(img)
    img_pil.save(path)

# -----------------------------
# Draw pose keypoints (optional)
# -----------------------------
def draw_pose(image, keypoints, color=(0, 255, 0)):
    """
    image: numpy BGR image
    keypoints: list of (x, y) tuples
    """
    for point in keypoints:
        cv2.circle(image, (int(point[0]), int(point[1])), 4, color, -1)
    return image
