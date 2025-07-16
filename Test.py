# test.py

from loader import VITONDataset
from torch.utils.data import DataLoader
from network import VITONHD
from utils import save_image

dataset = VITONDataset("./data")
loader = DataLoader(dataset, batch_size=1)

model = VITONHD()
# load .pth weights here

for person, cloth in loader:
    result = model.refiner(torch.cat([person, cloth], dim=1))
    save_image(result, "output.jpg")
    break
    


