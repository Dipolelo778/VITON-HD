# utils.py

import torch
from torchvision import transforms
from PIL import Image

def load_image(path):
    img = Image.open(path).convert("RGB")
    return img

def save_image(tensor, path):
    img = transforms.ToPILImage()(tensor.cpu().squeeze(0))
    img.save(path)

def denormalize(tensor):
    return tensor * 0.5 + 0.5
