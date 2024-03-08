import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import cv2

import mediapipe as mp

mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

POSE_CLASSES = ['pushup_up', 'pushup_down']

LANDMARK_PAIRS = {
    'wrists': (15, 16),
    'l_shoulder_wrist': (11, 15),
    'l_shoulder_elbow': (11, 13),
    'l_elbow_wrist': (13, 15),
    'r_shoulder_wrist': (12, 16),
    'r_shoulder_elbow': (12, 14),
    'r_elbow_wrist': (14, 16),
    'l_wrist_hip': (15, 23),
    'hips': (23, 24),
    'r_wrist_hip': (16, 24),
    'l_wrist_ankle': (15, 27),
    'l_shoulder_ankle': (11, 27),
    'r_wrist_ankle': (16, 28),
    'r_shoulder_ankle': (12, 28),
    'l_hip_ankle': (23, 27),
    'l_hip_knee': (23, 25),
    'l_knee_ankle': (25, 27),
    'r_hip_ankle': (24, 28),
    'r_hip_knee': (24, 26),
    'r_knee_ankle': (26, 28),
    'ankles': (27, 28),
}


def get_landmarks(pose, img):
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    landmarks = results.pose_landmarks
    world_landmarks = results.pose_world_landmarks
    for lmk in landmarks.landmark:
        lmk.visibility = 1
    for lmk in world_landmarks.landmark:
        lmk.visibility = 1
    return landmarks, world_landmarks


def calc_dist(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = b - a
    distance = np.sqrt(np.sum(diff ** 2))
    return distance


def get_features(lmks, hips=True):
    features = {}
    for pair in LANDMARK_PAIRS:
        a_n, b_n = LANDMARK_PAIRS[pair]
        a = lmks[a_n]
        b = lmks[b_n]
        dist = calc_dist(a, b)
        features[pair] = dist

    if hips:
        hips_dist = features['hips']
        ratio = 0.3 / hips_dist
        for pair in features:
            features[pair] *= ratio

    return features


def get_pose_data(pose_class, world=True, hips=True):
    img_path = f'img/{pose_class}'
    img_names = os.listdir(img_path)

    pose_data_headers = list(LANDMARK_PAIRS.keys())
    pose_data_headers.append('hash')
    pose_data = []

    with mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            model_complexity=2) as pose:

        for img_name in img_names:
            img = cv2.imread(f'{img_path}/{img_name}')
            img_hash, _ = img_name.split('.')

            try:
                _, world_landmarks = get_landmarks(pose, img)
            except Exception as e:
                print(f'{e} (hash: {img_hash})')
            else:
                lmks = np.asarray([[lmk.x, lmk.y, lmk.z] for lmk in world_landmarks.landmark])

                if not world:
                    height, width = img.shape[:2]
                    lmks *= [width, height, width]

                features = get_features(lmks, hips=hips)
                features['hash'] = img_hash
                pose_data.append(features)

    return pose_data_headers, pose_data


if __name__ == '__main__':
    if not os.path.isdir('data'):
        os.mkdir('data')
    for pose_class in POSE_CLASSES:
        pose_data_headers, pose_data = get_pose_data(pose_class)

        with open(f'data/{pose_class}.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=pose_data_headers)
            writer.writeheader()
            writer.writerows(pose_data)
