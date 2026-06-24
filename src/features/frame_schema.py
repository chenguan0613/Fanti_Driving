from dataclasses import dataclass


# Single-frame physical features
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


# Sliding window aggregation features
@dataclass
class WindowFeature:
    face_missing_ratio: float
    perclos: float
    blink_rate: float
    ear_mean: float
    ear_std: float
    ear_min: float
    mar_mean: float
    mar_std: float
    mar_max: float
    pitch_mean: float
    pitch_std: float
    gaze_x_mean: float
    gaze_y_mean: float


# Features used by AI
@dataclass
class FeatureRow:
    # Inherited from WindowFeature
    face_missing_ratio: float = 0.0
    perclos: float = 0.0
    blink_rate: float = 0.0
    ear_mean: float = 0.0
    ear_std: float = 0.0
    ear_min: float = 0.0
    mar_mean: float = 0.0
    mar_std: float = 0.0
    mar_max: float = 0.0
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    gaze_x_mean: float = 0.0
    gaze_y_mean: float = 0.0

    # Meta Data
    video_id: str = ""
    subject_id: str = ""
    label: int = -1

    # Enhanced Data
    ear_mean_norm: float = 0.0
    mar_max_norm: float = 0.0
    pitch_std_norm: float = 0.0
    ear_velocity: float = 0.0
    pitch_velocity: float = 0.0
    fatigue_index: float = 0.0
    mar_mean_norm: float = 0.0

    @staticmethod
    def from_window(stats: WindowFeature, video_id="", subject_id="", label=-1):
        return FeatureRow(
            face_missing_ratio=stats.face_missing_ratio,
            perclos=stats.perclos,
            blink_rate=stats.blink_rate,
            ear_mean=stats.ear_mean,
            ear_std=stats.ear_std,
            ear_min=stats.ear_min,
            mar_mean=stats.mar_mean,
            mar_std=stats.mar_std,
            mar_max=stats.mar_max,
            pitch_mean=stats.pitch_mean,
            pitch_std=stats.pitch_std,
            gaze_x_mean=stats.gaze_x_mean,
            gaze_y_mean=stats.gaze_y_mean,
            video_id=video_id,
            subject_id=subject_id,
            label=label,
        )


SMOOTH_COLS = [
    "perclos",
    "blink_rate",
    "ear_mean",
    "ear_std",
    "ear_min",
    "mar_mean",
    "mar_std",
    "mar_max",
    "pitch_mean",
    "pitch_std",
    "gaze_x_mean",
    "gaze_y_mean",
]
# Features required in the verification phase
NORM_COLS = [
    "ear_mean",
    "mar_mean",
    "mar_max",
    "pitch_std",
]
META_COLS = [
    "video_id",
    "subject_id",
    "label",
]
ENHANCED_COLS = [
    "ear_mean_norm",
    "mar_max_norm",
    "pitch_std_norm",
    "ear_velocity",
    "pitch_velocity",
    "fatigue_index",
]

# Features without cleaning
# RAW_FEATURES = [
#     "perclos",
#     "blink_rate",
#     "ear_mean",
#     "ear_std",
#     "ear_min",
#     "mar_mean",
#     "mar_std",
#     "mar_max",
#     "pitch_mean",
#     "pitch_std",
#     "gaze_x_mean",
#     "gaze_y_mean",
#     "ear_mean_norm",
#     "mar_max_norm",
#     "pitch_std_norm",
#     "ear_velocity",
#     "pitch_velocity",
#     "fatigue_index",
# ]

# Features used for training
GOLDEN_FEATURES = [
    "ear_std",
    "ear_min",
    "mar_std",
    "gaze_y_mean",
    "ear_mean_norm",
    "mar_mean_norm",
    "pitch_std_norm",
    "ear_velocity",
    "fatigue_index",
    "mar_max_norm",
    "perclos",
    "blink_rate",
]
