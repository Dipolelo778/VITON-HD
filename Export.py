# export.py

from PIL import Image
import torchvision.transforms as transforms

def save_image(tensor, path):
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor.cpu().detach().clamp(0, 1))
    image.save(path)
