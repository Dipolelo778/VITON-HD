import mediapipe as mp
import cv2
import numpy as np
import torch

mp_pose = mp.solutions.pose

class PoseEstimator:
    def __init__(self):
        self.pose = mp_pose.Pose(static_image_mode=True)

    def __call__(self, image_tensor):
        image_np = image_tensor.squeeze().permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
        results = self.pose.process(image_np)

        pose_map = np.zeros_like(image_np[..., 0], dtype=np.float32)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                cx, cy = int(lm.x * pose_map.shape[1]), int(lm.y * pose_map.shape[0])
                cv2.circle(pose_map, (cx, cy), 4, 1, -1)

        pose_map = torch.from_numpy(pose_map).unsqueeze(0).unsqueeze(0).float()
        return pose_map  # [1, 1, H, W]
