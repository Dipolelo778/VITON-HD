# test.py

import torch
from loader import VITONDataset
from torch.utils.data import DataLoader
from network import GeneratorUNet
from segmentation import SegmentationNet
from cloth_parser import ClothParser
from composite import composite
from export import save_image
from config import Config

def main():
    model = GeneratorUNet().to(Config.device)
    seg_net = SegmentationNet().to(Config.device)
    cloth_net = ClothParser().to(Config.device)

    model.load_state_dict(torch.load("checkpoints/model_final.pth"))
    model.eval()

    dataset = VITONDataset(Config.data_root)
    loader = DataLoader(dataset, batch_size=1)

    for i, (person, cloth) in enumerate(loader):
        person = person.to(Config.device)
        cloth = cloth.to(Config.device)

        seg_mask = seg_net(person)
        cloth_mask = cloth_net(cloth)

        output = model(torch.cat([person, cloth], dim=1))
        result = composite(person, output, seg_mask)

        save_image(result.squeeze(), f"results/output_{i}.png")

if __name__ == "__main__":
    main()
