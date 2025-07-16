import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from loader import VITONDataset
from segmentation import SegmentationNet
from pose_estimator import PoseEstimator
from cloth_parser import ClothParser
from gmm import GMM
from refiner import RefinerUNet
from inpainting import Inpainter
from config import Config

# ----------------------------
# Load dataset
# ----------------------------
dataset = VITONDataset(Config.DATA_ROOT)
loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

# ----------------------------
# Initialize models
# ----------------------------
segmentation = SegmentationNet().to(Config.DEVICE)
pose_estimator = PoseEstimator()  # CPU only
cloth_parser = ClothParser().to(Config.DEVICE)
gmm = GMM().to(Config.DEVICE)
refiner = RefinerUNet().to(Config.DEVICE)
inpainter = Inpainter().to(Config.DEVICE)

# ----------------------------
# Load checkpoints if exist
# ----------------------------
if os.path.exists(f"{Config.CHECKPOINTS}/segmentation.pth"):
    segmentation.load_state_dict(torch.load(f"{Config.CHECKPOINTS}/segmentation.pth"))
    print("✅ Loaded segmentation checkpoint")

if os.path.exists(f"{Config.CHECKPOINTS}/cloth_parser.pth"):
    cloth_parser.load_state_dict(torch.load(f"{Config.CHECKPOINTS}/cloth_parser.pth"))
    print("✅ Loaded ClothParser checkpoint")

if os.path.exists(f"{Config.CHECKPOINTS}/gmm.pth"):
    gmm.load_state_dict(torch.load(f"{Config.CHECKPOINTS}/gmm.pth"))
    print("✅ Loaded GMM checkpoint")

if os.path.exists(f"{Config.CHECKPOINTS}/refiner.pth"):
    refiner.load_state_dict(torch.load(f"{Config.CHECKPOINTS}/refiner.pth"))
    print("✅ Loaded Refiner checkpoint")

if os.path.exists(f"{Config.CHECKPOINTS}/inpainting.pth"):
    inpainter.load_state_dict(torch.load(f"{Config.CHECKPOINTS}/inpainting.pth"))
    print("✅ Loaded Inpainter checkpoint")

# ----------------------------
# Optimizer (PoseEstimator NOT trainable)
# ----------------------------
optimizer = optim.Adam(
    list(segmentation.parameters()) +
    list(cloth_parser.parameters()) +
    list(gmm.parameters()) +
    list(refiner.parameters()) +
    list(inpainter.parameters()),
    lr=Config.LR
)

loss_fn = nn.L1Loss()

# ----------------------------
# Training loop
# ----------------------------
for epoch in range(Config.EPOCHS):
    for person, cloth in loader:
        person = person.to(Config.DEVICE)
        cloth = cloth.to(Config.DEVICE)

        # Inference steps
        mask = segmentation(person)
        pose = pose_estimator(person.cpu()).to(Config.DEVICE)
        cloth_mask = cloth_parser(cloth)

        warped_cloth = gmm(person, pose, cloth_mask, cloth)
        refined = refiner(torch.cat([person, warped_cloth], dim=1))
        final_output = inpainter(refined)

        loss = loss_fn(final_output, person)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"✅ Epoch {epoch+1}/{Config.EPOCHS} - Loss: {loss.item():.4f}")

    if (epoch + 1) % 5 == 0:
        torch.save(segmentation.state_dict(), f"{Config.CHECKPOINTS}/segmentation.pth")
        torch.save(cloth_parser.state_dict(), f"{Config.CHECKPOINTS}/cloth_parser.pth")
        torch.save(gmm.state_dict(), f"{Config.CHECKPOINTS}/gmm.pth")
        torch.save(refiner.state_dict(), f"{Config.CHECKPOINTS}/refiner.pth")
        torch.save(inpainter.state_dict(), f"{Config.CHECKPOINTS}/inpainting.pth")
        print(f"💾 Saved checkpoints at epoch {epoch+1}")




