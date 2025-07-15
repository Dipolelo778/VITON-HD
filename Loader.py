# loader.py

import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class VITONDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.person_images = sorted(os.listdir(os.path.join(data_root, 'person')))
        self.cloth_images = sorted(os.listdir(os.path.join(data_root, 'cloth')))
        self.data_root = data_root
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 192)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.person_images)

    def __getitem__(self, idx):
        person_path = os.path.join(self.data_root, 'person', self.person_images[idx])
        cloth_path = os.path.join(self.data_root, 'cloth', self.cloth_images[idx % len(self.cloth_images)])

        person = Image.open(person_path).convert("RGB")
        cloth = Image.open(cloth_path).convert("RGB")

        person = self.transform(person)
        cloth = self.transform(cloth)

        return person, cloth
