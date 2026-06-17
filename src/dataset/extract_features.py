import argparse
import pandas as pd
from pathlib import Path
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from functools import partial
from src.preprocessing.video_loader import VideoLoader
from src.preprocessing.frame_extractor import FrameExtractor
from src.realtime.buffer import SlidingWindowBuffer
from src.features.window_agg import WindowAggregator


def process_video(row, window_stride=15):
    video_id = row["video_id"]
    video_path = row["video_path"]
    state_label = row["state"]
    subject_label = row.get("subject_id", f"subject_{video_id}")

    print(f"[START] {video_id}")

    extractor = FrameExtractor("./face_landmarker.task")
    loader = VideoLoader(video_path, target_fps=30)
    window_buffer = SlidingWindowBuffer(window_size=90)

    all_features = []
    frame_counter = 0

    for frame_id, timestamp, frame in loader.frame():
        feature = extractor.extract(frame, timestamp=timestamp)
        window_buffer.push(feature)

        if window_buffer.is_ready():
            if frame_counter % window_stride == 0:
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


def _append_video(features: list[dict], path: Path):
    if not features:
        return
    df = pd.DataFrame(features)
    id_cols = ["video_id", "subject_id", "label"]
    cols = [c for c in df.columns if c not in id_cols] + id_cols
    df = df[cols]
    write_header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=write_header)


def main():
    total_cores = cpu_count()

    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=None, help="Part index (0-based)")
    parser.add_argument(
        "--total-parts", type=int, default=None, help="Total number of parts"
    )
    parser.add_argument("--window-stride", type=int, default=15)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Worker processes (default: total cores / parts, or {total_cores - 1})",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="src/dataset/dataset.csv",
        help="Input index CSV",
    )
    args = parser.parse_args()

    index_csv_path = Path(args.input)
    output_features_path = Path("src/dataset/training_features.csv")

    index_df = pd.read_csv(index_csv_path)
    rows = [row for _, row in index_df.iterrows()]

    if args.total_parts is not None and args.part is not None:
        chunk_size = max(1, len(rows) // args.total_parts)
        start = args.part * chunk_size
        end = start + chunk_size if args.part < args.total_parts - 1 else len(rows)
        rows = rows[start:end]
        output_features_path = output_features_path.with_stem(
            f"training_features_part{args.part}"
        )
        print(
            f"Part {args.part}/{args.total_parts}: videos {start}-{end-1} -> {output_features_path}"
        )

    parts = args.total_parts or 1
    workers = args.workers if args.workers is not None else max(1, total_cores // parts)
    print(f"Using {workers} worker processes")

    with Pool(processes=workers) as pool:
        worker = partial(process_video, window_stride=args.window_stride)
        for features in pool.imap_unordered(worker, rows):
            _append_video(features, output_features_path)


if __name__ == "__main__":
    main()
