import os
import torch
from torchvision import transforms
from PIL import Image

from segmentation import BodySegmentation
from pose_estimator import PoseEstimator
from gmm import GMM
from refiner import Refiner
from inpainting import Inpainter

# ----------------------------
# CONFIG
# ----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS = {
    "segmentation": "./checkpoints/segmentation.pth",
    "pose": "./checkpoints/pose_estimator.pth",
    "gmm": "./checkpoints/gmm.pth",
    "refiner": "./checkpoints/refiner.pth",
    "inpainting": "./checkpoints/inpainting.pth"
}

INPUT_DIR = "./test_images"
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# LOAD MODELS
# ----------------------------

segmentation = BodySegmentation().to(DEVICE)
segmentation.load_state_dict(torch.load(CHECKPOINTS["segmentation"], map_location=DEVICE))
segmentation.eval()

pose_estimator = PoseEstimator().to(DEVICE)
pose_estimator.load_state_dict(torch.load(CHECKPOINTS["pose"], map_location=DEVICE))
pose_estimator.eval()

gmm = GMM().to(DEVICE)
gmm.load_state_dict(torch.load(CHECKPOINTS["gmm"], map_location=DEVICE))
gmm.eval()

refiner = Refiner().to(DEVICE)
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

# ----------------------------
# RUN PIPELINE
# ----------------------------

for img_name in os.listdir(INPUT_DIR):
    if not img_name.endswith(('.jpg', '.png')):
        continue

    img_path = os.path.join(INPUT_DIR, img_name)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        seg_mask = segmentation(img_tensor)
        pose = pose_estimator(img_tensor)
        warped_cloth = gmm(img_tensor, pose, seg_mask)
        refined = refiner(img_tensor, warped_cloth)
        final_output = inpainter(refined)

    output_img = transforms.ToPILImage()(final_output.squeeze().cpu())
    output_img.save(os.path.join(OUTPUT_DIR, f"result_{img_name}"))

print(f"✅ Done! Results saved to: {OUTPUT_DIR}")
    
