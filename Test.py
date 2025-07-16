import os
import torch
from torchvision import transforms
from PIL import Image
from config import Config
from loader import load_image_tensor
from pose_estimator import PoseEstimator
from cloth_parser import ClothParser
from segmentation import Segmentation
from gmm import GeometricMatchingModule
from refiner import Refiner
from utils import save_output_image

# -----------------------------
# 1️⃣ Setup
# -----------------------------
device = Config.device

# Load models
pose_estimator = PoseEstimator().to(device)
cloth_parser = ClothParser().to(device)
segmenter = Segmentation().to(device)
gmm = GeometricMatchingModule().to(device)
refiner = Refiner().to(device)

# Load trained weights
checkpoint = torch.load(Config.test_checkpoint, map_location=device)
pose_estimator.load_state_dict(checkpoint['pose_estimator'])
cloth_parser.load_state_dict(checkpoint['cloth_parser'])
segmenter.load_state_dict(checkpoint['segmenter'])
gmm.load_state_dict(checkpoint['gmm'])
refiner.load_state_dict(checkpoint['refiner'])

pose_estimator.eval()
cloth_parser.eval()
segmenter.eval()
gmm.eval()
refiner.eval()

# -----------------------------
# 2️⃣ Input Images
# -----------------------------
transform = transforms.Compose([
    transforms.Resize(Config.img_size),
    transforms.ToTensor(),
])

person_img = load_image_tensor(Config.test_person_img, transform).unsqueeze(0).to(device)
cloth_img = load_image_tensor(Config.test_cloth_img, transform).unsqueeze(0).to(device)

# -----------------------------
# 3️⃣ Pipeline
# -----------------------------
with torch.no_grad():
    cloth_mask = cloth_parser(cloth_img)
    pose_map = pose_estimator(person_img)
    seg_mask = segmenter(person_img)
    warped_cloth = gmm(cloth_img, cloth_mask, pose_map)
    output = refiner(person_img, warped_cloth, seg_mask)

# -----------------------------
# 4️⃣ Save Result
# -----------------------------
save_output_image(output, Config.output_dir, "tryon_result.png")

print(f"✅ Try-On done! Saved at {Config.output_dir}/tryon_result.png")
    


