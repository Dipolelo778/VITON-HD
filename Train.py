import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from network import UNetResNet


# -------------------------
# 1️⃣  CONFIG
# -------------------------

class Config:
    data_root = "./data"   # Folder with 'person' and 'clothes'
    epochs = 50
    batch_size = 4
    lr = 2e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)


# -------------------------
# 2️⃣  DATASET
# -------------------------

class VITONDataset(Dataset):
    def __init__(self, root, transform=None):
        self.persons = sorted(os.listdir(os.path.join(root, "person")))
        self.clothes = sorted(os.listdir(os.path.join(root, "clothes")))
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.persons)

    def __getitem__(self, idx):
        person_path = os.path.join(self.root, "person", self.persons[idx])
        cloth_path = os.path.join(self.root, "clothes", self.clothes[idx % len(self.clothes)])  # repeat if needed

        person_img = Image.open(person_path).convert("RGB")
        cloth_img = Image.open(cloth_path).convert("RGB")

        if self.transform:
            person_img = self.transform(person_img)
            cloth_img = self.transform(cloth_img)

        # Input is person + cloth (6 channels)
        input = torch.cat([person_img, cloth_img], dim=0)
        return input, person_img  # input, target


# -------------------------
# 3️⃣  TRAINING LOGIC
# -------------------------

def train():
    transform = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
    ])

    dataset = VITONDataset(Config.data_root, transform)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

    model = UNetResNet().to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.L1Loss()

    for epoch in range(Config.epochs):
        model.train()
        epoch_loss = 0

        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(Config.device)
            targets = targets.to(Config.device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{Config.epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}] finished with avg loss: {epoch_loss/len(dataloader):.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(Config.save_dir, f"model_epoch_{epoch+1}.pth"))

    print("Training done!")


# -------------------------
# 4️⃣  ENTRY POINT
# -------------------------

if __name__ == "__main__":
    train()tr
