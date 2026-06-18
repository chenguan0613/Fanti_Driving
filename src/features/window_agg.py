import numpy as np
from typing import List
from src.features.frame_schema import FrameFeature, WindowFeature


class WindowAggregator:
    @staticmethod
    def aggregate(frames: List[FrameFeature]) -> WindowFeature:
        if not frames:
            return WindowFeature(
                face_missing_ratio=1.0,
                perclos=0.0,
                blink_rate=0.0,
                ear_mean=0.0,
                ear_std=0.0,
                ear_min=0.0,
                mar_mean=0.0,
                mar_std=0.0,
                mar_max=0.0,
                pitch_mean=0.0,
                pitch_std=0.0,
                yaw_mean=0.0,
                yaw_std=0.0,
                gaze_x_mean=0.0,
                gaze_y_mean=0.0,
            )

        total_frames = len(frames)
        valid_frames = [f for f in frames if f.face_detected == 1]
        if not valid_frames:
            return WindowFeature(
                face_missing_ratio=1.0,
                perclos=0.0,
                blink_rate=0.0,
                ear_mean=0.0,
                ear_std=0.0,
                ear_min=0.0,
                mar_mean=0.0,
                mar_std=0.0,
                mar_max=0.0,
                pitch_mean=0.0,
                pitch_std=0.0,
                yaw_mean=0.0,
                yaw_std=0.0,
                gaze_x_mean=0.0,
                gaze_y_mean=0.0,
            )

        face_missing_ratio = 1.0 - (len(valid_frames) / total_frames)

        # Extracting numerical sequences
        ears = [(f.left_EAR + f.right_EAR) / 2.0 for f in valid_frames]
        mars = [f.mouth_open_ratio for f in valid_frames]
        pitches = [f.head_pitch for f in valid_frames]
        yaws = [f.head_yaw for f in valid_frames]
        gaze_xs = [f.gaze_x for f in valid_frames]
        gaze_ys = [f.gaze_y for f in valid_frames]

        # calculate PERCLOS
        closed_frames_count = sum([f.eye_closed for f in valid_frames])
        perclos = closed_frames_count / len(valid_frames)

        # detect blinks from raw EAR waveform
        # a blink is a rapid drop below 60% of median EAR, then recovery
        baseline = np.median(ears)
        threshold = baseline * 0.6

        blink_count = 0
        in_blink = False
        for ear in ears:
            if ear < threshold:
                if not in_blink:
                    blink_count += 1
                    in_blink = True
            else:
                in_blink = False

        window_duration = valid_frames[-1].timestamp - valid_frames[0].timestamp
        blink_rate = blink_count / window_duration if window_duration > 0 else 0.0

        return WindowFeature(
            face_missing_ratio=face_missing_ratio,
            perclos=perclos,
            blink_rate=blink_rate,
            ear_mean=float(np.mean(ears)),
            ear_std=float(np.std(ears)),
            ear_min=float(np.min(ears)),
            mar_mean=float(np.mean(mars)),
            mar_std=float(np.std(mars)),
            mar_max=float(np.max(mars)),
            pitch_mean=float(np.mean(pitches)),
            pitch_std=float(np.std(pitches)),
            yaw_mean=float(np.mean(yaws)),
            yaw_std=float(np.std(yaws)),
            gaze_x_mean=float(np.mean(gaze_xs)),
            gaze_y_mean=float(np.mean(gaze_ys)),
        )
