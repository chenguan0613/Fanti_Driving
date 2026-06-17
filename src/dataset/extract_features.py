import os
import pandas as pd
from pathlib import Path
from dataclasses import asdict
from src.preprocessing.video_loader import VideoLoader
from src.preprocessing.frame_extractor import FrameExtractor
from src.realtime.buffer import SlidingWindowBuffer
from src.features.window_agg import WindowAggregator


def main():
    index_csv_path = Path("src/dataset/dataset.csv")
    output_features_path = Path("src/dataset/training_features.csv")

    index_df = pd.read_csv(index_csv_path)
    extractor = FrameExtractor("./face_landmarker.task")
    all_features = []
    for index, row in index_df.iterrows():
        video_id = row["video_id"]
        video_path = row["video_path"]
        state_label = row["state"]
        print(
            f"\nProcessing [{index+1}/{len(index_df)}]: ID={video_id}, State={state_label}"
        )
        loader = VideoLoader(video_path, target_fps=30)
        window_buffer = SlidingWindowBuffer(window_size=90)
        frame_count = 0
        window_count = 0
        for (
            frame_id,
            timestamp,
            frame,
        ) in loader.frame():
            frame_count += 1
            feature = extractor.extract(frame, timestamp=timestamp)
            window_buffer.push(feature)
            if window_buffer.is_ready():
                stats_obj = WindowAggregator.aggregate(window_buffer.get_window())
                # 人脸丢失率低于50% in 3 seconds
                if stats_obj and stats_obj.face_missing_ratio < 0.50:
                    row_data = asdict(stats_obj)
                    row_data["video_id"] = video_id
                    row_data["label"] = state_label
                    all_features.append(row_data)
                    window_count += 1
    if not all_features:
        print("\n Cannot Extract")
        return
    final_df = pd.DataFrame(all_features)
    cols = [c for c in final_df.columns if c not in ["video_id", "label"]] + [
        "video_id",
        "label",
    ]
    final_df = final_df[cols]
    final_df.to_csv(output_features_path, index=False)
    extractor.close()


if __name__ == "__main__":
    main()
