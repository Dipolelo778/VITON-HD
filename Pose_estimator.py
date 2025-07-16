# pose_estimator.py

import cv2
import mediapipe as mp
import numpy as np

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True)

    def get_keypoints(self, image_path):
        """
        Takes image path → returns normalized keypoints array.
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        keypoints = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            keypoints = np.array(keypoints)
        else:
            keypoints = np.zeros((33, 4))  # 33 pose keypoints

        return keypoints

if __name__ == "__main__":
    pe = PoseEstimator()
    kpts = pe.get_keypoints("./sample_person.jpg")
    print(kpts)
