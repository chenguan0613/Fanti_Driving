from collections import deque
from pathlib import Path
import sys
import time
import winsound

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
LOCAL_LIBS = ROOT / "python_libs"
if LOCAL_LIBS.exists():
    sys.path.insert(0, str(LOCAL_LIBS))

# MediaPipe may try to import TensorFlow only for optional documentation helpers.
# This project does not use TensorFlow, and skipping it avoids protobuf conflicts.
sys.modules["tensorflow"] = None

import mediapipe as mp


MODELS_DIR = ROOT / "models"

EYE_MODEL_PATH = MODELS_DIR / "eye_model.pkl"
YAWN_MODEL_PATH = MODELS_DIR / "yawn_model.pkl"

UNSAFE_SECONDS = 2.0
POSSIBLY_UNSAFE_SECONDS = 0.7
WARNING_COOLDOWN_SECONDS = 2.0


LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]


def build_hog_descriptor(image_size):
    return cv2.HOGDescriptor(
        _winSize=tuple(image_size),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )


def extract_hog_from_crop(crop, image_size):
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, tuple(image_size))
    resized = cv2.equalizeHist(resized)
    hog = build_hog_descriptor(tuple(image_size))
    return hog.compute(resized).reshape(1, -1)


def landmark_to_point(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)


def crop_region(frame, points, margin=20):
    height, width = frame.shape[:2]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(min(xs) - margin, 0)
    y1 = max(min(ys) - margin, 0)
    x2 = min(max(xs) + margin, width)
    y2 = min(max(ys) + margin, height)
    return frame[y1:y2, x1:x2]


def crop_eye(frame, face_landmarks):
    height, width = frame.shape[:2]
    points = []
    for index in LEFT_EYE + RIGHT_EYE:
        points.append(landmark_to_point(face_landmarks.landmark[index], width, height))
    return crop_region(frame, points, margin=25)


def crop_face(frame, face_landmarks):
    height, width = frame.shape[:2]
    points = [
        landmark_to_point(landmark, width, height)
        for landmark in face_landmarks.landmark
    ]
    return crop_region(frame, points, margin=10)


def predict_crop(model_bundle, crop):
    features = extract_hog_from_crop(crop, model_bundle["image_size"])
    if features is None:
        return None
    return model_bundle["model"].predict(features)[0]


def update_timer(condition, timer, frame_time):
    return timer + frame_time if condition else 0.0


def decide_status(eye_closed_time, yawn_time, hand_time):
    max_risk_time = max(eye_closed_time, yawn_time, hand_time)
    if max_risk_time >= UNSAFE_SECONDS:
        return "Unsafe"
    if max_risk_time >= POSSIBLY_UNSAFE_SECONDS:
        return "Possibly Unsafe"
    return "Safe"


def draw_label(frame, text, position, color):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    if not EYE_MODEL_PATH.exists() or not YAWN_MODEL_PATH.exists():
        raise FileNotFoundError("Run train_models.py before realtime_detection.py.")

    eye_model = joblib.load(EYE_MODEL_PATH)
    yawn_model = joblib.load(YAWN_MODEL_PATH)

    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    eye_closed_time = 0.0
    yawn_time = 0.0
    hand_time = 0.0
    last_warning_time = 0.0
    last_frame_time = time.time()
    status_history = deque(maxlen=5)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh, mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            frame_time = now - last_frame_time
            last_frame_time = now

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb)
            hand_results = hands.process(rgb)

            eye_state = "Unknown"
            mouth_state = "Unknown"

            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                eye_crop = crop_eye(frame, face_landmarks)
                face_crop = crop_face(frame, face_landmarks)

                eye_state = predict_crop(eye_model, eye_crop) or "Unknown"
                mouth_state = predict_crop(yawn_model, face_crop) or "Unknown"

            hand_present = bool(hand_results.multi_hand_landmarks)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )

            eye_closed_time = update_timer(eye_state == "Closed", eye_closed_time, frame_time)
            yawn_time = update_timer(mouth_state == "yawn", yawn_time, frame_time)
            hand_time = update_timer(hand_present, hand_time, frame_time)

            status_history.append(decide_status(eye_closed_time, yawn_time, hand_time))
            if "Unsafe" in status_history:
                status = "Unsafe"
            elif "Possibly Unsafe" in status_history:
                status = "Possibly Unsafe"
            else:
                status = "Safe"

            if status == "Unsafe" and now - last_warning_time > WARNING_COOLDOWN_SECONDS:
                winsound.Beep(1200, 250)
                last_warning_time = now

            color = (0, 200, 0)
            if status == "Possibly Unsafe":
                color = (0, 180, 255)
            elif status == "Unsafe":
                color = (0, 0, 255)

            draw_label(frame, f"Eye: {eye_state}", (20, 35), (255, 255, 255))
            draw_label(frame, f"Mouth: {mouth_state}", (20, 70), (255, 255, 255))
            draw_label(frame, f"Hand: {'Detected' if hand_present else 'Not detected'}", (20, 105), (255, 255, 255))
            draw_label(frame, f"Status: {status}", (20, 145), color)

            cv2.imshow("Dangerous Driving Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
