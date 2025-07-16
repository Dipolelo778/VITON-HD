# pose_estimator.py

import mediapipe as mp
import cv2
import json

mp_pose = mp.solutions.pose

def extract_pose(image_path, output_json):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)
        keypoints = []

        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                keypoints.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                })

    with open(output_json, "w") as f:
        json.dump(keypoints, f)
