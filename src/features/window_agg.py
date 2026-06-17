import numpy as np
from typing import List
from src.features.frame_schema import FrameFeature, WindowFeature


class WindowAggregator:
    @staticmethod
    def aggregate(frames: List[FrameFeature]) -> dict:
        if not frames:
            return {}
        total_frames = len(frames)
        valid_frames = [f for f in frames if f.face_detected == 1]
        if not valid_frames:
            return WindowFeature(
                face_missing_ratio=1.0,
                perclos=0.0,
                ear_mean=0.0,
                ear_std=0.0,
                mar_mean=0.0,
                mar_max=0.0,
                pitch_mean=0.0,
                pitch_std=0.0,
                yaw_std=0.0,
            )

        face_missing_ratio = 1.0 - (len(valid_frames) / total_frames)

        # Extracting numerical sequences
        ears = [(f.left_EAR + f.right_EAR) / 2.0 for f in valid_frames]
        mars = [f.mouth_open_ratio for f in valid_frames]
        pitches = [f.head_pitch for f in valid_frames]
        yaws = [f.head_yaw for f in valid_frames]

        # calculate PERCLOS
        closed_frames_count = sum([f.eye_closed for f in valid_frames])
        perclos = closed_frames_count / len(valid_frames)

        return WindowFeature(
            face_missing_ratio=face_missing_ratio,
            perclos=perclos,
            ear_mean=float(np.mean(ears)),
            ear_std=float(np.std(ears)),
            mar_mean=float(np.mean(mars)),
            mar_max=float(np.max(mars)),
            pitch_mean=float(np.mean(pitches)),
            pitch_std=float(np.std(pitches)),
            yaw_std=float(np.std(yaws)),
        )
