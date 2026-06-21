import pandas as pd


# eye: perclos, blink_rate, ear_mean, ear_std, ear_min
# mouth: mar_mean, mar_std, mar_max
# head: pitch_mean, pitch_std, yaw_mean, yaw_std
# gaze: gaze_x_mean, gaze_y_mean
class FeatureBaseline:
    def __init__(self, input_path: str):
        self.df = pd.read_csv(input_path)
        # Delete sliding windows with a frame drop rate > 50%.
        initial_len = len(self.df)
        self.df = self.df[self.df["face_missing_ratio"] < 0.50].reset_index(drop=True)
        self.df.drop(columns=["face_missing_ratio"], inplace=True)

    def _smooth_noise(self):
        print("Noise reduction in progress...")
        smooth_cols = [
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
            "yaw_mean",
            "yaw_std",
            "gaze_x_mean",
            "gaze_y_mean",
        ]
        self.df = self.df.sort_values(by=["video_id"]).reset_index(drop=True)
        for col in smooth_cols:
            self.df[col] = self.df.groupby("video_id")[col].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )

    def _normalize_baseline(self):
        print("Normalizing...")
        awake_df = self.df[self.df["label"] == 0]
        # Select features need to be eliminate difference
        norm_cols = ["ear_mean", "mar_max", "pitch_std", "yaw_std"]
        baseline_table = awake_df.groupby("subject_id")[norm_cols].mean()
        for col in norm_cols:
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
        meta_cols = {"video_id", "subject_id", "label"}
        features = [c for c in self.df.columns if c not in meta_cols]
