import numpy as np
from typing import List
from src.features.frame_schema import FrameFeature


class WindowAggregator:
    @staticmethod
    def aggregate(frames: List[FrameFeature]) -> dict:
        if not frames:
            return {}
        total_frames = len(frames)
        valid_frames = [f for f in frames if f.face_detected == 1]
        if not valid_frames:
            return {"face_missing_ratio": 1.0}

        face_missing_ratio = 1.0 - (len(valid_frames) / total_frames)

        # Extracting numerical sequences
        ears = [(f.left_EAR + f.right_EAR) / 2.0 for f in valid_frames]
        mars = [f.mouth_open_ratio for f in valid_frames]
        pitches = [f.head_pitch for f in valid_frames]
        yaws = [f.head_yaw for f in valid_frames]

        # calculate PERCLOS
        closed_frames_count = sum([f.eye_closed for f in valid_frames])
        perclos = closed_frames_count / len(valid_frames)

        return {
            "face_missing_ratio": face_missing_ratio,
            "perclos": perclos,
            # Mean eye opening and tremor
            "ear_mean": np.mean(ears),
            "ear_std": np.std(ears),
            # Mouth features
            "mar_mean": np.mean(mars),
            "mar_max": np.max(mars),
            # Head pitch characteristics
            "pitch_mean": np.mean(pitches),
            "pitch_std": np.std(pitches),
            # Head yaw characteristics
            "yaw_std": np.std(yaws),
        }
