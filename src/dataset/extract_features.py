import pandas as pd
from pathlib import Path
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from functools import partial
from src.preprocessing import VideoLoader, FrameExtractor
from src.realtime import SlidingWindowBuffer
from src.features import WindowAggregator, FeatureRow, META_COLS


def _process_chunk(rows, window_stride, target_fps, window_size, model_path):
    extractor = FrameExtractor(model_path)
    results = []
    try:
        for row in rows:
            # Parse the metadata in the basic video index table
            video_id = row["video_id"]
            video_path = row["video_path"]
            state_label = row["state"]
            subject_label = row.get("subject_id", f"subject_{video_id}")

            print(f"[START] {video_id}")
            # Create the sliding window
            loader = VideoLoader(video_path, target_fps=target_fps)
            window_buffer = SlidingWindowBuffer(window_size=window_size)

            all_features = []
            frame_counter = 0

            try:
                # Load the video frame by frame
                for _, timestamp, frame in loader.frame():
                    feature = extractor.extract(frame, timestamp=timestamp)
                    window_buffer.push(feature)
                    # If the sliding window is full, start analyzing
                    if window_buffer.is_ready():
                        if frame_counter % window_stride == 0:
                            # Calculate macro characteristics
                            stats = WindowAggregator.aggregate(
                                window_buffer.get_window()
                            )

                            if stats and stats.face_missing_ratio < 0.50:
                                # Convert the object into one-dimensional structured data rows
                                row = FeatureRow.from_window(
                                    stats, video_id, subject_label, state_label
                                )
                                all_features.append(row)
                        frame_counter += 1
            except Exception as e:
                print(f"[ERROR] {video_id}: {e}")

            print(f"[DONE] {video_id}, windows={len(all_features)}")

            results.append(all_features)
    finally:
        extractor.close()

    return results


class FeatureExtractor:
    def __init__(
        self,
        window_stride: int,
        target_fps: int,
        window_size: int,
        model_path: str = "./face_landmarker.task",
    ) -> None:
        self.window_stride = window_stride
        self.target_fps = target_fps
        self.window_size = window_size
        self.model_path = model_path

    def _append_video(self, features: list[FeatureRow], path: Path):
        if not features:
            return
        rows = [asdict(f) for f in features]
        df = pd.DataFrame(rows)
        cols = [c for c in df.columns if c not in META_COLS] + list(META_COLS)
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

        n_chunks = max(1, min(len(rows), workers * 2))
        chunk_size = (len(rows) + n_chunks - 1) // n_chunks
        chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]

        with Pool(processes=workers) as pool:
            worker = partial(
                _process_chunk,
                window_stride=self.window_stride,
                target_fps=self.target_fps,
                window_size=self.window_size,
                model_path=self.model_path,
            )
            for chunk_features in pool.imap_unordered(worker, chunks):
                for features in chunk_features:
                    self._append_video(features, output_features_path)
