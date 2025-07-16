import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class VITONDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.person_dir = os.path.join(data_root, "person")
        self.cloth_dir = os.path.join(data_root, "cloth")
        self.pose_dir = os.path.join(data_root, "pose")
        self.mask_dir = os.path.join(data_root, "cloth_mask")

        self.person_files = sorted(os.listdir(self.person_dir))
        self.cloth_files = sorted(os.listdir(self.cloth_dir))

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 192)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.person_files)

    def __getitem__(self, idx):
        person_path = os.path.join(self.person_dir, self.person_files[idx])
        cloth_path = os.path.join(self.cloth_dir, self.cloth_files[idx % len(self.cloth_files)])  # loop
        pose_path = os.path.join(self.pose_dir, self.person_files[idx].replace('.jpg', '_pose.png'))
        mask_path = os.path.join(self.mask_dir, self.cloth_files[idx % len(self.cloth_files)].replace('.jpg', '_mask.png'))

        person = Image.open(person_path).convert("RGB")
        cloth = Image.open(cloth_path).convert("RGB")
        pose = Image.open(pose_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        person = self.transform(person)
        cloth = self.transform(cloth)
        pose = self.transform(pose)
        mask = self.transform(mask)

        inputs = {
            "person": person,     # [3,H,W]
            "cloth": cloth,       # [3,H,W]
            "pose": pose,         # [1,H,W]
            "mask": mask          # [1,H,W]
        }

        return inputs

def get_dataloader(data_root, batch_size=4, shuffle=True):
    dataset = VITONDataset(data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
