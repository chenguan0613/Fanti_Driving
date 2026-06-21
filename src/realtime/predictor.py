from collections import deque
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for lib_dir in (PROJECT_ROOT / "python_libs", WORKSPACE_ROOT / "python_libs"):
    if lib_dir.exists():
        sys.path.insert(0, str(lib_dir))

# MediaPipe can try to import TensorFlow for optional helpers. The project does
# not use TensorFlow, and skipping it avoids protobuf conflicts on this machine.
sys.modules["tensorflow"] = None

import cv2
import mediapipe as mp
import numpy as np


class FatiguePredictor:
    def __init__(
        self,
        eye_model_path=None,
        yawn_model_path=None,
        unsafe_seconds=5.0,
        possible_seconds=3.0,
        ear_threshold=0.18,
        mar_threshold=0.60,
        possible_yawns_per_minute=2,
        unsafe_yawns_per_minute=3,
    ):
        print("[INFO] Loading FANTI_DRIVING monitoring core...")

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.drawing = mp.solutions.drawing_utils
        self.hand_connections = mp.solutions.hands.HAND_CONNECTIONS

        self.unsafe_seconds = unsafe_seconds
        self.possible_seconds = possible_seconds
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.possible_yawns_per_minute = possible_yawns_per_minute
        self.unsafe_yawns_per_minute = unsafe_yawns_per_minute

        self.current_status = "Initializing"
        self.fatigue_prob = 0.0
        self.perclos = 0.0
        self.fps = 0.0

        self.eye_state = "Unknown"
        self.mouth_state = "Unknown"
        self.hand_state = "Not detected"
        self.hand_count = 0
        self.ear = 0.0
        self.mar = 0.0

        self.eye_closed_time = 0.0
        self.hand_time = 0.0
        self.was_yawning = False
        self.yawn_events = deque()
        self.prev_time = time.time()
        self.eye_history = deque(maxlen=90)

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1024, 768))

        now = time.time()
        frame_time = max(now - self.prev_time, 1e-6)
        self.prev_time = now
        self.fps = 1.0 / frame_time

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb)
        hand_results = self.hands.process(rgb)

        self.eye_state = "Unknown"
        self.mouth_state = "Unknown"

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            self.ear = self._compute_ear(face_landmarks)
            self.mar = self._compute_mar(face_landmarks)
            self.eye_state = "Closed" if self.ear < self.ear_threshold else "Open"
            self.mouth_state = "yawn" if self.mar > self.mar_threshold else "no_yawn"
        else:
            self.current_status = "No Face Detected"

        self.hand_count = len(hand_results.multi_hand_landmarks or [])
        hand_present = self.hand_count > 0
        self.hand_state = f"{self.hand_count} hand(s)" if hand_present else "Not detected"
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.hand_connections,
                )

        self.eye_closed_time = self._update_timer(
            self.eye_state == "Closed", self.eye_closed_time, frame_time
        )
        self.hand_time = self._update_timer(hand_present, self.hand_time, frame_time)
        self._update_yawn_events(now, self.mouth_state == "yawn")

        self.eye_history.append(1 if self.eye_state == "Closed" else 0)
        self.perclos = sum(self.eye_history) / len(self.eye_history)

        if face_results.multi_face_landmarks:
            self.current_status = self._decide_status()

        self.fatigue_prob = self._risk_score()
        annotated_frame = self._draw_hud(frame)
        return annotated_frame, self.current_status, self.fatigue_prob

    def _compute_ear(self, face_landmarks):
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        def eye_aspect_ratio(indexes):
            p = lambda i: self._landmark_array(face_landmarks, i)
            vertical_1 = np.linalg.norm(p(indexes[1]) - p(indexes[5]))
            vertical_2 = np.linalg.norm(p(indexes[2]) - p(indexes[4]))
            horizontal = np.linalg.norm(p(indexes[0]) - p(indexes[3]))
            return float((vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6))

        return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

    def _compute_mar(self, face_landmarks):
        p = lambda i: self._landmark_array(face_landmarks, i)
        vertical_1 = np.linalg.norm(p(13) - p(14))
        vertical_2 = np.linalg.norm(p(82) - p(87))
        vertical_3 = np.linalg.norm(p(312) - p(317))
        horizontal = np.linalg.norm(p(61) - p(291))
        return float((vertical_1 + vertical_2 + vertical_3) / (3.0 * horizontal + 1e-6))

    def _landmark_array(self, face_landmarks, index):
        landmark = face_landmarks.landmark[index]
        return np.array([landmark.x, landmark.y], dtype=np.float32)

    def _landmark_points(self, frame, face_landmarks, indexes):
        height, width = frame.shape[:2]
        points = []
        for index in indexes:
            landmark = face_landmarks.landmark[index]
            points.append((int(landmark.x * width), int(landmark.y * height)))
        return points

    def _crop_region(self, frame, points, margin):
        height, width = frame.shape[:2]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x1 = max(min(xs) - margin, 0)
        y1 = max(min(ys) - margin, 0)
        x2 = min(max(xs) + margin, width)
        y2 = min(max(ys) + margin, height)
        return frame[y1:y2, x1:x2]

    def _update_timer(self, condition, timer, frame_time):
        return timer + frame_time if condition else 0.0

    def _update_yawn_events(self, now, is_yawning):
        if is_yawning and not self.was_yawning:
            self.yawn_events.append(now)
        self.was_yawning = is_yawning

        while self.yawn_events and now - self.yawn_events[0] > 60.0:
            self.yawn_events.popleft()

    def _decide_status(self):
        possible_count = 0

        if self.eye_closed_time >= self.unsafe_seconds:
            return "Unsafe"
        if self.eye_closed_time >= self.possible_seconds:
            possible_count += 1

        if len(self.yawn_events) >= self.unsafe_yawns_per_minute:
            return "Unsafe"
        if len(self.yawn_events) >= self.possible_yawns_per_minute:
            possible_count += 1

        if self.hand_count >= 2:
            return "Unsafe"
        if self.hand_count == 1 and self.hand_time >= self.unsafe_seconds:
            return "Unsafe"
        if self.hand_count == 1 and self.hand_time >= self.possible_seconds:
            possible_count += 1

        if possible_count >= 2:
            return "Unsafe"
        if possible_count == 1:
            return "Possibly Unsafe"
        return "Safe"

    def _risk_score(self):
        score = 0.0
        score += min(self.eye_closed_time / self.unsafe_seconds, 1.0) * 40.0
        score += min(len(self.yawn_events) / self.unsafe_yawns_per_minute, 1.0) * 35.0
        score += min(self.hand_time / self.unsafe_seconds, 1.0) * 25.0
        if self.hand_count >= 2:
            score = max(score, 80.0)
        return round(score, 1)

    def _draw_hud(self, frame):
        display_frame = frame.copy()

        color = (0, 255, 0)
        if self.current_status == "Possibly Unsafe":
            color = (0, 180, 255)
        elif self.current_status == "Unsafe":
            color = (0, 0, 255)
        elif self.current_status == "No Face Detected":
            color = (0, 255, 255)

        self._draw_text(display_frame, f"FPS: {self.fps:.1f}", (20, 30), (255, 255, 0))
        self._draw_text(display_frame, f"Eye: {self.eye_state}", (20, 70), (255, 255, 255))
        self._draw_text(display_frame, f"EAR: {self.ear:.3f}", (20, 105), (255, 255, 255))
        self._draw_text(display_frame, f"Mouth: {self.mouth_state}", (20, 140), (255, 255, 255))
        self._draw_text(display_frame, f"MAR: {self.mar:.3f}", (20, 175), (255, 255, 255))
        self._draw_text(display_frame, f"Yawns/min: {len(self.yawn_events)}", (20, 210), (255, 255, 255))
        self._draw_text(display_frame, f"Hand: {self.hand_state}", (20, 245), (255, 255, 255))
        self._draw_text(display_frame, f"Status: {self.current_status}", (20, 290), color, 1.0, 3)
        self._draw_text(display_frame, f"PERCLOS: {self.perclos * 100:.1f}%", (20, 330), color)
        self._draw_text(display_frame, f"Risk Score: {self.fatigue_prob:.1f}%", (20, 365), color)

        if self.current_status == "Unsafe":
            self._draw_text(
                display_frame,
                "WARNING: DANGEROUS DRIVING DETECTED",
                (20, 720),
                (0, 0, 255),
                1.0,
                3,
            )

        return display_frame

    def _draw_text(self, frame, text, position, color, scale=0.75, thickness=2):
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def close(self):
        self.face_mesh.close()
        self.hands.close()
