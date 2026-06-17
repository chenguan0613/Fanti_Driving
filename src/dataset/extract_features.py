import pandas as pd
from pathlib import Path
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from src.preprocessing.video_loader import VideoLoader
from src.preprocessing.frame_extractor import FrameExtractor
from src.realtime.buffer import SlidingWindowBuffer
from src.features.window_agg import WindowAggregator


def process_video(row):
    video_id = row["video_id"]
    video_path = row["video_path"]
    state_label = row["state"]

    print(f"[START] {video_id}")

    extractor = FrameExtractor("./face_landmarker.task")
    loader = VideoLoader(video_path, target_fps=30)
    window_buffer = SlidingWindowBuffer(window_size=90)

    all_features = []

    for frame_id, timestamp, frame in loader.frame():
        feature = extractor.extract(frame, timestamp=timestamp)
        window_buffer.push(feature)

        if window_buffer.is_ready():
            stats = WindowAggregator.aggregate(window_buffer.get_window())

            if stats and stats.face_missing_ratio < 0.50:
                row_data = asdict(stats)
                row_data["video_id"] = video_id
                row_data["label"] = state_label
                all_features.append(row_data)

    extractor.close()

    print(f"[DONE] {video_id}, windows={len(all_features)}")

    return all_features


def main():
    index_csv_path = Path("src/dataset/dataset.csv")
    output_features_path = Path("src/dataset/training_features.csv")

    index_df = pd.read_csv(index_csv_path)
    rows = [row for _, row in index_df.iterrows()]

    workers = max(1, cpu_count() - 1)

    print(f"Using {workers} processes")

    with Pool(processes=workers) as pool:
        results = pool.map(process_video, rows)

    # flatten results
    all_features = []
    for r in results:
        all_features.extend(r)

    if not all_features:
        print("Cannot Extract Features")
        return

    final_df = pd.DataFrame(all_features)

    cols = [c for c in final_df.columns if c not in ["video_id", "label"]] + [
        "video_id",
        "label",
    ]

    final_df = final_df[cols]

    final_df.to_csv(output_features_path, index=False)

    print(f"Saved to {output_features_path}")


if __name__ == "__main__":
    main()
