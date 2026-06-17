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


@dataclass
class WindowFeature:
    face_missing_ratio: float
    perclos: float
    ear_mean: float
    ear_std: float
    ear_min: float
    mar_mean: float
    mar_std: float
    mar_max: float
    pitch_mean: float
    pitch_std: float
    yaw_mean: float
    yaw_std: float
    gaze_x_mean: float
    gaze_y_mean: float
