import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import Config
from loader import VITONDataset
from pose_estimator import PoseEstimator
from cloth_parser import ClothParser
from segmentation import Segmentation
from gmm import GeometricMatchingModule
from refiner import Refiner
from utils import VGGLoss

# -----------------------------
# 1️⃣  Setup
# -----------------------------
device = Config.device

# Networks
pose_estimator = PoseEstimator().to(device)
cloth_parser = ClothParser().to(device)
segmenter = Segmentation().to(device)
gmm = GeometricMatchingModule().to(device)
refiner = Refiner().to(device)

# Loss functions
l1_loss = nn.L1Loss()
vgg_loss = VGGLoss()

# Optimizers
optimizer = optim.Adam(
    list(pose_estimator.parameters()) +
    list(cloth_parser.parameters()) +
    list(segmenter.parameters()) +
    list(gmm.parameters()) +
    list(refiner.parameters()),
    lr=Config.lr
)

# -----------------------------
# 2️⃣  Dataset
# -----------------------------
transform = transforms.Compose([
    transforms.Resize(Config.img_size),
    transforms.ToTensor(),
])

dataset = VITONDataset(Config.data_root, transform)
dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

# -----------------------------
# 3️⃣  Training Loop
# -----------------------------
for epoch in range(Config.epochs):
    epoch_loss = 0
    for i, batch in enumerate(dataloader):
        person_img = batch['person'].to(device)
        cloth_img = batch['cloth'].to(device)

        # (1) Parse cloth mask
        cloth_mask = cloth_parser(cloth_img)

        # (2) Estimate pose
        pose_map = pose_estimator(person_img)

        # (3) Segment body parts
        seg_mask = segmenter(person_img)

        # (4) Warp cloth
        warped_cloth = gmm(cloth_img, cloth_mask, pose_map)

        # (5) Refine final try-on
        output = refiner(person_img, warped_cloth, seg_mask)

        # Loss
        l1 = l1_loss(output, person_img)
        vgg = vgg_loss(output, person_img)
        loss = l1 + Config.vgg_weight * vgg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch [{epoch+1}/{Config.epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}] finished. Avg Loss: {epoch_loss/len(dataloader):.4f}")

    # Save checkpoints
    save_path = os.path.join(Config.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
    torch.save({
        'pose_estimator': pose_estimator.state_dict(),
        'cloth_parser': cloth_parser.state_dict(),
        'segmenter': segmenter.state_dict(),
        'gmm': gmm.state_dict(),
        'refiner': refiner.state_dict(),
    }, save_path)

print("✅ Training finished!")



