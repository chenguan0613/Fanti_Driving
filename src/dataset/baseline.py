import pandas as pd
from src.features import SMOOTH_COLS, NORM_COLS


class FeatureBaseline:
    def __init__(self, input_path: str):
        self.df = pd.read_csv(input_path)
        # Delete sliding windows with a frame drop rate > 50%.
        self.df = self.df[self.df["face_missing_ratio"] < 0.50].reset_index(drop=True)
        self.df.drop(columns=["face_missing_ratio"], inplace=True)

    def _smooth_noise(self):
        print("Noise reduction in progress...")
        self.df = self.df.sort_values(by=["video_id"]).reset_index(drop=True)
        for col in SMOOTH_COLS:
            self.df[col] = self.df.groupby("video_id")[col].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )

    def _normalize_baseline(self):
        print("Normalizing...")
        awake_df = self.df[self.df["label"] == 0]
        baseline_table = awake_df.groupby("subject_id")[NORM_COLS].mean()
        for col in NORM_COLS:
            baseline_series = self.df["subject_id"].map(baseline_table[col])
            if baseline_series.isnull().any():
                global_mean = self.df.groupby("subject_id")[col].mean()
                baseline_series = baseline_series.fillna(
                    self.df["subject_id"].map(global_mean)
                )
            if col == "mar_max":
                self.df[f"{col}_norm"] = self.df[col] - baseline_series
            else:
                self.df[f"{col}_norm"] = (self.df[col] - baseline_series) / (
                    baseline_series + 1e-6
                )

    def _calculate_dynamics(self):
        self.df["ear_velocity"] = (
            self.df.groupby("video_id")["ear_mean_norm"].diff().fillna(0.0)
        )
        self.df["pitch_velocity"] = (
            self.df.groupby("video_id")["pitch_std_norm"].diff().fillna(0.0)
        )
        self.df["fatigue_index"] = self.df["perclos"] * self.df["ear_mean_norm"].abs()

    def process_and_save(self, output_path: str):
        self._smooth_noise()
        self._normalize_baseline()
        self._calculate_dynamics()
        self.df.to_csv(output_path, index=False)
