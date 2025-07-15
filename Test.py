import os
import torch
from torchvision import transforms
from PIL import Image
from network import UNetResNet

# -------------------------
# 1️⃣  CONFIG
# -------------------------

class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "./checkpoints/model_epoch_50.pth"  # Adjust epoch if needed
    output_dir = "./outputs"
    test_data = "./data/test"  # folder with 'person' and 'clothes'

    os.makedirs(output_dir, exist_ok=True)


# -------------------------
# 2️⃣  LOAD MODEL
# -------------------------

def load_model():
    model = UNetResNet().to(Config.device)
    model.load_state_dict(torch.load(Config.checkpoint_path, map_location=Config.device))
    model.eval()
    return model


# -------------------------
# 3️⃣  TEST DATASET LOADER
# -------------------------

def load_images(person_path, cloth_path):
    transform = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
    ])

    person = Image.open(person_path).convert("RGB")
    cloth = Image.open(cloth_path).convert("RGB")

    person = transform(person)
    cloth = transform(cloth)

    input = torch.cat([person, cloth], dim=0).unsqueeze(0)  # Add batch dim

    return input.to(Config.device)


# -------------------------
# 4️⃣  INFERENCE LOGIC
# -------------------------

def test():
    model = load_model()

    persons = sorted(os.listdir(os.path.join(Config.test_data, "person")))
    clothes = sorted(os.listdir(os.path.join(Config.test_data, "clothes")))

    for idx, person_name in enumerate(persons):
        person_path = os.path.join(Config.test_data, "person", person_name)
        cloth_path = os.path.join(Config.test_data, "clothes", clothes[idx % len(clothes)])  # cycle clothes

        input = load_images(person_path, cloth_path)

        with torch.no_grad():
            output = model(input).squeeze(0).cpu()

        output_img = transforms.ToPILImage()(output)
        save_path = os.path.join(Config.output_dir, f"tryon_{idx}.png")
        output_img.save(save_path)

        print(f"Saved: {save_path}")

    print("✅ Testing complete! Check your outputs folder.")


# -------------------------
# 5️⃣  ENTRY POINT
# -------------------------

if __name__ == "__main__":
    test()
