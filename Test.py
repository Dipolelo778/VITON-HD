# test.py

import torch
from loader import load_image, save_image
from cloth_parser import ClothParser
from segmentation import HumanParser
from pose_estimator import PoseEstimator
from gmm import GeometricMatchingModule
from refiner import RefinerUNet

import config

def main():
    # -----------------------
    # Load models
    # -----------------------
    cloth_parser = ClothParser().to(config.device).eval()
    human_parser = HumanParser().to(config.device).eval()
    pose_estimator = PoseEstimator().to(config.device).eval()
    gmm = GeometricMatchingModule().to(config.device).eval()
    refiner = RefinerUNet().to(config.device).eval()

    # -----------------------
    # Load input images
    # -----------------------
    person_img = load_image("inputs/person.jpg").to(config.device)  # [1, 3, H, W]
    cloth_img = load_image("inputs/cloth.jpg").to(config.device)    # [1, 3, H, W]

    # -----------------------
    # Step 1: Parse cloth
    # -----------------------
    cloth_mask = cloth_parser(cloth_img)

    # -----------------------
    # Step 2: Parse person
    # -----------------------
    person_mask = human_parser(person_img)

    # -----------------------
    # Step 3: Detect pose
    # -----------------------
    pose = pose_estimator(person_img)

    # -----------------------
    # Step 4: Warp cloth with GMM
    # -----------------------
    warped_cloth = gmm(cloth_img, cloth_mask, pose)

    # -----------------------
    # Step 5: Refine try-on output
    # -----------------------
    output = refiner(person_img, warped_cloth, person_mask)

    # -----------------------
    # Save result
    # -----------------------
    save_image(output, "outputs/tryon_result.jpg")

    print("✅ Try-On complete → saved to outputs/tryon_result.jpg")

if __name__ == "__main__":
    main()
    

