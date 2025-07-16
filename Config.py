# config.py

import os

class Config:
    # DATA
    DATA_ROOT = "./data"
    PERSON_DIR = os.path.join(DATA_ROOT, "person")
    CLOTH_DIR = os.path.join(DATA_ROOT, "cloth")
    SEG_MODEL_PATH = "./checkpoints/segmentation.pth"
    GMM_MODEL_PATH = "./checkpoints/gmm.pth"
    REFINER_MODEL_PATH = "./checkpoints/refiner.pth"

    # TRAINING
    EPOCHS = 50
    BATCH_SIZE = 4
    LR = 2e-4
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    SAVE_DIR = "./checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # IMAGE
    IMG_HEIGHT = 256
    IMG_WIDTH = 192

    # LOSS
    VGG_PATH = "./checkpoints/vgg19-dcbb9e9d.pth"
