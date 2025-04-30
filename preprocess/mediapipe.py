import cv2
import numpy as np
import mediapipe as mp

HANDS_MP_IDX = [i for i in range(21)]
RIGHT_HANDS_IDX = [i for i in range(len(HANDS_MP_IDX))]
LEFT_HANDS_IDX = [i + len(RIGHT_HANDS_IDX) for i in range(len(HANDS_MP_IDX))]
HANDS_IDX = [i for i in range(len(RIGHT_HANDS_IDX) + len(LEFT_HANDS_IDX))]

HEAD = 0
RIGHT_SHOULDER = 12
LEFT_SHOULDER = 11
RIGHT_ELBOW = 14
LEFT_ELBOW = 13
RIGHT_WRIST = 16
LEFT_WRIST = 15

BODY_MP_IDX = [HEAD, RIGHT_SHOULDER, LEFT_SHOULDER, RIGHT_ELBOW, LEFT_ELBOW, RIGHT_WRIST, LEFT_WRIST]
BODY_IDX = [idx + 2*len(HANDS_MP_IDX) for idx in range(len(BODY_MP_IDX))]
LEFT_ARM_IDX = [0, 2, 4, 6]
RIGHT_ARM_IDX = [0, 1, 3, 5]

BODY_MP_IDX_TO_BODY_IDX = {mp_idx: idx + 2*len(HANDS_MP_IDX) for idx, mp_idx in enumerate(BODY_MP_IDX)}

TOTAL_LANDMARKS = len(BODY_MP_IDX) + 2*len(HANDS_MP_IDX)

class MP_LandmarkExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


    def extract_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_pose = self.pose.process(image_rgb)
        results_hands = self.hands.process(image_rgb)

        landmarks = np.zeros((TOTAL_LANDMARKS, 3))

        # Pose landmarks (shoulders, elbows, wrists)
        if results_pose.pose_landmarks:
            for mp_idx, idx in BODY_MP_IDX_TO_BODY_IDX.items():
                lm = results_pose.pose_landmarks.landmark[mp_idx]
                landmarks[idx] = np.array([lm.x, lm.y, lm.z])

        # Hand landmarks
        if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
            for hand_landmarks, handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                hand_label = handedness.classification[0].label
                for i, lm in enumerate(hand_landmarks.landmark):
                    if hand_label == "Right":
                        idx = RIGHT_HANDS_IDX[i]
                    else:
                        idx = LEFT_HANDS_IDX[i]
                    landmarks[idx] = np.array([lm.x, lm.y, lm.z])

        return landmarks

    def get_landmark_from_path(self, video_path):
        cap = cv2.VideoCapture(video_path)
        landmarks_sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = self.extract_landmarks(frame)
            landmarks_sequence.append(landmarks)

        cap.release()
        return np.array(landmarks_sequence)  # Shape: (num_frames, num_features)