import cv2
import time
import numpy as np
import joblib
import traceback
import pandas as pd
from collections import deque
from dataclasses import asdict
from src.preprocessing import FrameExtractor
from .buffer import SlidingWindowBuffer
from src.features import WindowAggregator, FeatureRow, NORM_COLS


class FatiguePredictor:
    def __init__(
        # self, model_path="models/fatigue_model.pkl", task_path="face_landmarker.task"
        self,
        model_path="models/heuristic_model.pkl",
        task_path="face_landmarker.task",
        window_size=150,
    ):
        # Load the features
        loaded_obj = joblib.load(model_path)
        if isinstance(loaded_obj, dict) and "model" in loaded_obj:
            self.model = loaded_obj["model"]
        else:
            self.model = loaded_obj

        if isinstance(loaded_obj, dict) and "feature_names" in loaded_obj:
            self.active_features = list(loaded_obj["feature_names"])
            print(
                f"Read the feature list from the model dictionary ({len(self.active_features)} dimensions)."
            )
        elif hasattr(self.model, "feature_names_in_"):
            self.active_features = list(self.model.feature_names_in_)
            print(
                f"Read the feature list from inside the model ({len(self.active_features)} dimensions)."
            )
        else:
            raise ValueError("Unable to obtain feature list")

        # Initialization of component and state
        self.extractor = FrameExtractor(task_path)
        self.buffer = SlidingWindowBuffer(window_size=window_size)

        self.current_status = "Initializing"
        self.fatigue_prob = 0.0
        self.perclos = 0.0
        self.fps = 0
        self.prev_time = 0

        self.is_baseline_ready = False
        self.baseline_history = []
        self.baseline_stats = {}

        self.norm_history = deque(maxlen=15)

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1024, 768))

        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = current_time

        feature = self.extractor.extract(frame, timestamp=current_time)
        self.buffer.push(feature)

        if self.buffer.is_ready():
            raw_stats = WindowAggregator.aggregate(self.buffer.get_window())
            if raw_stats and raw_stats.face_missing_ratio < 0.50:
                self.perclos = raw_stats.perclos
                self._think(raw_stats)
            else:
                self.current_status = "No Face Detected"
        else:
            self.current_status = "Buffering"

        annotated_frame = self._draw_hud(frame)
        return annotated_frame, self.current_status, self.fatigue_prob

    def _think(self, raw_stats):
        try:
            row = FeatureRow.from_window(raw_stats)

            # Phase 1: Build baseline
            if not self.is_baseline_ready:
                self.baseline_history.append(asdict(row))
                self.current_status = "Calibrating"
                if len(self.baseline_history) >= 150:
                    df_base = pd.DataFrame(self.baseline_history)
                    for feat in NORM_COLS:
                        self.baseline_stats[feat] = df_base[feat].mean()
                    self.is_baseline_ready = True
                    print(
                        "\nPersonal baseline calibration complete. Real-time monitoring now in progress!\n"
                    )
                return

            # Phase 2: calculate the enhanced features
            for feat in ["ear_mean", "pitch_std", "yaw_std"]:
                base_val = self.baseline_stats[feat]
                curr_val = getattr(row, feat)
                setattr(row, f"{feat}_norm", (curr_val - base_val) / (base_val + 1e-6))

            row.mar_max_norm = row.mar_max - self.baseline_stats["mar_max"]

            curr_ear_norm = row.ear_mean_norm
            curr_pitch_norm = row.pitch_std_norm

            if len(self.norm_history) == 15:
                old_ear, old_pitch = self.norm_history[0]
                row.ear_velocity = curr_ear_norm - old_ear
                row.pitch_velocity = curr_pitch_norm - old_pitch
            else:
                row.ear_velocity = 0.0
                row.pitch_velocity = 0.0

            self.norm_history.append((curr_ear_norm, curr_pitch_norm))
            row.fatigue_index = self.perclos * abs(curr_ear_norm)

            # Phase 3: AI prediction
            input_values = [getattr(row, col, 0.0) for col in self.active_features]
            X_input = np.array([input_values])

            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(X_input)[0]
                self.fatigue_prob = round(probabilities[1] * 100, 1)
            else:
                decision = self.model.decision_function(X_input)[0]
                prob = 1.0 / (1.0 + np.exp(-decision))
                self.fatigue_prob = round(prob * 100, 1)

            if self.fatigue_prob > 70.0:
                self.current_status = "FATIGUE WARNING"
            else:
                self.current_status = "Safe"

        except Exception as e:
            print(f"\n[ERROR] AI inference failed: {str(e)}")
            traceback.print_exc()
            print("--------------------------------------------------\n")
            self.current_status = "Error: Check Terminal"

    def _draw_hud(self, frame):
        display_frame = frame.copy()
        if self.current_status == "FATIGUE WARNING":
            hud_color = (0, 0, 255)
            cv2.putText(
                display_frame,
                "WARNING: FATIGUE DETECTED!",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )
        elif self.current_status == "Safe":
            hud_color = (0, 255, 0)
        elif self.current_status == "Calibrating":
            hud_color = (255, 255, 0)
            cv2.putText(
                display_frame,
                "CALIBRATING BASELINE...",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                hud_color,
                3,
            )
        else:
            hud_color = (0, 255, 255)
        if self.current_status in ["Safe", "FATIGUE WARNING"]:
            cv2.putText(
                display_frame,
                f"PERCLOS: {self.perclos*100:.1f}%",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                hud_color,
                2,
            )
            cv2.putText(
                display_frame,
                f"AI Prob: {self.fatigue_prob}%",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                hud_color,
                2,
            )
        cv2.putText(
            display_frame,
            f"FPS: {self.fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        return display_frame

    def close(self):
        self.extractor.close()
