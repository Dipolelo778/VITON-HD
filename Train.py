# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loader import VITONDataset
from network import GeneratorUNet
from config import Config

def main():
    dataset = VITONDataset(Config.data_root)
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

    model = GeneratorUNet().to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.L1Loss()

    for epoch in range(Config.epochs):
        for i, (person, cloth) in enumerate(loader):
            person = person.to(Config.device)
            cloth = cloth.to(Config.device)

            input = torch.cat([person, cloth], dim=1)
            output = model(input)

            loss = criterion(output, person)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{Config.epochs}], Step [{i}], Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), f"{Config.save_dir}/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()



