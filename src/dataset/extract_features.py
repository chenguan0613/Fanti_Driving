import pandas as pd
from pathlib import Path
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from functools import partial
from src.preprocessing import VideoLoader, FrameExtractor
from src.realtime import SlidingWindowBuffer
from src.features import WindowAggregator


class FeatureExtractor:
    """
    FeatureExtractor extracts raw features from a given dataset.
    """

    def __init__(
        self,
        window_stride: int,
        target_fps: int,
        window_size: int,
    ) -> None:
        self.window_stride = window_stride
        self.target_fps = target_fps
        self.window_size = window_size

    def process_video(self, row):
        video_id = row["video_id"]
        video_path = row["video_path"]
        state_label = row["state"]
        subject_label = row.get("subject_id", f"subject_{video_id}")

        print(f"[START] {video_id}")

        extractor = FrameExtractor("./face_landmarker.task")
        loader = VideoLoader(video_path, target_fps=self.target_fps)
        window_buffer = SlidingWindowBuffer(window_size=self.window_size)

        all_features = []
        frame_counter = 0

        for _, timestamp, frame in loader.frame():
            feature = extractor.extract(frame, timestamp=timestamp)
            window_buffer.push(feature)

            if window_buffer.is_ready():
                if frame_counter % self.window_stride == 0:
                    stats = WindowAggregator.aggregate(window_buffer.get_window())

                    if stats and stats.face_missing_ratio < 0.50:
                        row_data = asdict(stats)
                        row_data["video_id"] = video_id
                        row_data["subject_id"] = subject_label
                        row_data["label"] = state_label
                        all_features.append(row_data)
                frame_counter += 1

        extractor.close()

        print(f"[DONE] {video_id}, windows={len(all_features)}")

        return all_features

    def _append_video(self, features: list[dict], path: Path):
        if not features:
            return
        df = pd.DataFrame(features)
        id_cols = ["video_id", "subject_id", "label"]
        cols = [c for c in df.columns if c not in id_cols] + id_cols
        df = df[cols]
        write_header = not path.exists()
        df.to_csv(path, mode="a", index=False, header=write_header)

    def extract(
        self,
        dataset_list_path: str,
        output_path: str,
        workers: int = cpu_count(),
    ):
        index_csv_path = Path(dataset_list_path)
        output_features_path = Path(output_path)

        index_df = pd.read_csv(index_csv_path)
        rows = [row for _, row in index_df.iterrows()]

        print(f"Using {workers} worker processes")

        with Pool(processes=workers) as pool:
            worker = partial(self.process_video)
            for features in pool.imap_unordered(worker, rows):
                self._append_video(features, output_features_path)
