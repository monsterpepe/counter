import cv2
import numpy as np
from classifier import PoseClassifier
from data import mp_pose, POSE_CLASSES, get_landmarks, get_features


if __name__ == '__main__':
    pc = PoseClassifier(5, POSE_CLASSES)
    pc.fit()
    cap = cv2.VideoCapture(f'test/vid.mp4')
    result = None

    dist_threshold = 0.35
    prev_pose = None
    count = 0

    if not cap.isOpened():
        print('Error opening video')

    with mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            model_complexity=1) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if result is None:
                    height, width = frame.shape[:2]
                    result = cv2.VideoWriter(
                        f'test/result.mp4',
                        cv2.VideoWriter_fourcc(*'MP4V'),
                        30, (width, height))

                landmarks, world_landmarks = get_landmarks(pose, frame)
                lmks = np.asarray([[lmk.x, lmk.y, lmk.z] for lmk in world_landmarks.landmark])
                features = get_features(lmks).values()
                pred_pose = pc.predict(features)
                dist_mean = pc.get_dist_means(features)

                if pred_pose != prev_pose and dist_mean < dist_threshold:
                    if pred_pose == 'pushup_up' and prev_pose is not None:
                        count += 1
                    prev_pose = pred_pose

                annot_frame = frame.copy()
                annot_frame = cv2.putText(annot_frame, pred_pose, (0, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
                annot_frame = cv2.putText(annot_frame, str(dist_mean), (0, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
                annot_frame = cv2.putText(annot_frame, f'Count: {count}', (0, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
                result.write(annot_frame)
            else:
                break

    cap.release()
    result.release()
