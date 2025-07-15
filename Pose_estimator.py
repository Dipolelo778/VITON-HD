# pose_estimator.py

import cv2
import numpy as np

class DummyPoseEstimator:
    def __init__(self):
        pass

    def estimate(self, image_path):
        # You should plug real OpenPose/DensePose
        return np.zeros((18, 2))  # Dummy keypoints
