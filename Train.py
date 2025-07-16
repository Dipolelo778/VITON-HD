# train.py

import torch
from torch import nn, optim
from loader import VITONDataset
from torch.utils.data import DataLoader
from network import VITONHD
from config import Config

dataset = VITONDataset(Config.DATA_ROOT)
loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

model = VITONHD()
optimizer = optim.Adam([
    {'params': model.segmentation.parameters()},
    {'params': model.cloth_parser.parameters()},
    {'params': model.gmm.parameters()},
    {'params': model.refiner.parameters()},
    {'params': model.inpainter.parameters()},
], lr=Config.LR)

loss_fn = nn.L1Loss()

for epoch in range(Config.EPOCHS):
    for person, cloth in loader:
        output = model.refiner(torch.cat([person, cloth], dim=1))
        loss = loss_fn(output, person)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()




