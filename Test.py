import torch
from torchvision import transforms
from PIL import Image
import os

from segmentation import SegmentationNet
from pose_estimator import PoseEstimator
from gmm import GMM
from refiner import RefinerUNet
from inpainting import Inpainter

# ----------------------------
# CONFIG
# ----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Use Google Drive if running in Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

USE_GOOGLE_DRIVE = True  # Set to False for local dev

# NOTE: If you're in Colab, mount your Drive in a notebook cell:
# from google.colab import drive
# drive.mount('/content/drive')

if USE_GOOGLE_DRIVE and IN_COLAB:
    ROOT_PATH = "/content/drive/MyDrive/VITON_HD"
else:
    ROOT_PATH = "."  # Local path

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

person = Image.open(INPUT_PERSON).convert("RGB")
cloth = Image.open(INPUT_CLOTH).convert("RGB")

person_tensor = transform(person).unsqueeze(0).to(DEVICE)
cloth_tensor = transform(cloth).unsqueeze(0).to(DEVICE)

# ----------------------------
# INFERENCE
# ----------------------------

with torch.no_grad():
    mask = segmentation(person_tensor)
    pose = pose_estimator(person_tensor.cpu())
    pose = pose.to(DEVICE)

    warped_cloth, _ = gmm(person_tensor, pose, mask, cloth_tensor)
    refined = refiner(torch.cat([person_tensor, warped_cloth], dim=1))
    final_output = inpainter(refined)

# ----------------------------
# SAVE RESULT
# ----------------------------

result_img = transforms.ToPILImage()(final_output.squeeze().cpu())
result_img.save(OUTPUT_PATH)

print(f"✅ Saved result at: {OUTPUT_PATH}")
    




