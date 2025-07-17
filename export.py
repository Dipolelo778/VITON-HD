# export.py
import os
import torch
from torchvision import transforms
from PIL import Image

from segmentation import SegmentationNet
from pose_estimator import PoseEstimator
from gmm import GMM
from refiner import RefinerUNet
from inpainting import Inpainter

# ----------------------------
# ✅ CONFIG: add Drive toggle
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_GOOGLE_DRIVE = True  # 👈 True = training in Colab, False = deployed app

if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT_PATH = "/content/drive/MyDrive/VITON_HD"
else:
    ROOT_PATH = "."

CHECKPOINTS = {
    "segmentation": os.path.join(ROOT_PATH, "checkpoints/segmentation.pth"),
    "gmm": os.path.join(ROOT_PATH, "checkpoints/gmm.pth"),
    "refiner": os.path.join(ROOT_PATH, "checkpoints/refiner.pth"),
    "inpainting": os.path.join(ROOT_PATH, "checkpoints/inpainting.pth")
}

INPUT_PERSON = os.path.join(ROOT_PATH, "test_images/person.jpg")
INPUT_CLOTH = os.path.join(ROOT_PATH, "test_images/cloth.jpg")
OUTPUT_PATH = os.path.join(ROOT_PATH, "results/final_tryon.jpg")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ----------------------------
# LOAD MODELS
# ----------------------------

segmentation = SegmentationNet().to(DEVICE)
segmentation.load_state_dict(torch.load(CHECKPOINTS["segmentation"], map_location=DEVICE))
segmentation.eval()

pose_estimator = PoseEstimator()

gmm = GMM().to(DEVICE)
gmm.load_state_dict(torch.load(CHECKPOINTS["gmm"], map_location=DEVICE))
gmm.eval()

refiner = RefinerUNet().to(DEVICE)
refiner.load_state_dict(torch.load(CHECKPOINTS["refiner"], map_location=DEVICE))
refiner.eval()

inpainter = Inpainter().to(DEVICE)
inpainter.load_state_dict(torch.load(CHECKPOINTS["inpainting"], map_location=DEVICE))
inpainter.eval()

# ----------------------------
# IMAGE PREP
# ----------------------------

transform = transforms.Compose([
    transforms.Resize((256, 192)),
    transforms.ToTensor(),
])

person_img = Image.open(INPUT_PERSON).convert("RGB")
cloth_img = Image.open(INPUT_CLOTH).convert("RGB")

person_tensor = transform(person_img).unsqueeze(0).to(DEVICE)
cloth_tensor = transform(cloth_img).unsqueeze(0).to(DEVICE)

# ----------------------------
# RUN PIPELINE
# ----------------------------

with torch.no_grad():
    seg_mask = segmentation(person_tensor)
    pose = pose_estimator(person_tensor.cpu())  # Mediapipe runs on CPU
    pose = pose.to(DEVICE)

    warped_cloth, _ = gmm(person_tensor, pose, seg_mask, cloth_tensor)
    refined = refiner(torch.cat([person_tensor, warped_cloth], dim=1))
    final_output = inpainter(refined)

output_img = transforms.ToPILImage()(final_output.squeeze().cpu())
output_img.save(OUTPUT_PATH)

print(f"✅ Done! Virtual try-on saved at: {OUTPUT_PATH}")
    


