from dataclasses import dataclass


@dataclass
class FrameFeature:
    timestamp: float
    face_detected: int
    eye_closed: int = 0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    left_EAR: float = 0.0
    mouth_open_ratio: float = 0.0
    raw_landmarks: list | None = None
    right_EAR: float = 0.0
