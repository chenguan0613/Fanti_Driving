from dataclasses import dataclass


@dataclass
class FrameFeature:
    timestamp: float
    face_detected: int
    left_EAR: float = 0
    right_EAR: float = 0
    eye_closed: int = 0
    mounth_open_ratio: float = 0
    head_pitch: float = 0
    head_yaw: float = 0
    gaze_x: float = 0
    gaze_y: float = 0
