# loader.py

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class VITONDataset(Dataset):
    def __init__(self, root, transform=None):
        self.persons = sorted(os.listdir(os.path.join(root, "person")))
        self.clothes = sorted(os.listdir(os.path.join(root, "clothes")))
        self.root = root
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 192)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.persons)

    def __getitem__(self, idx):
        person_path = os.path.join(self.root, "person", self.persons[idx])
        cloth_path = os.path.join(self.root, "clothes", self.clothes[idx % len(self.clothes)])

        person_img = Image.open(person_path).convert("RGB")
        cloth_img = Image.open(cloth_path).convert("RGB")

        person = self.transform(person_img)
        cloth = self.transform(cloth_img)

        return person, cloth
