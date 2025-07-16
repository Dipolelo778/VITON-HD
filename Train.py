# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from loader import VITONDataset
from torch.utils.data import DataLoader

from segmentation import SegmentationNet
from pose_estimator import PoseEstimator
from gmm import GMM
from refiner import RefinerUNet
from inpainting import Inpainter

from config import Config

dataset = VITONDataset(Config.DATA_ROOT)
loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

segmentation = SegmentationNet().to(Config.DEVICE)
pose_estimator = PoseEstimator()
gmm = GMM().to(Config.DEVICE)
refiner = RefinerUNet().to(Config.DEVICE)
inpainter = Inpainter().to(Config.DEVICE)

optimizer = optim.Adam(
    list(segmentation.parameters()) +
    list(gmm.parameters()) +
    list(refiner.parameters()) +
    list(inpainter.parameters()),
    lr=Config.LR
)

l1_loss = nn.L1Loss()

for epoch in range(Config.EPOCHS):
    for person, cloth in loader:
        person = person.to(Config.DEVICE)
        cloth = cloth.to(Config.DEVICE)

        mask = segmentation(person)
        pose = pose_estimator(person)
        pose = pose.to(Config.DEVICE)

        warped_cloth = gmm(person, pose, mask, cloth)
        refined = refiner(torch.cat([person, warped_cloth], dim=1))
        final_output = inpainter(refined)

        loss = l1_loss(final_output, person)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} Loss: {loss.item()}")




