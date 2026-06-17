import csv
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_CSV = Path("src/dataset/dataset.csv")


def get_state(filename: str) -> int:
    stem = Path(filename).stem
    return 0 if stem in ("0", "5") else 1


def main():
    rows = []
    video_id = 1

    for fold_dir in sorted(DATA_DIR.iterdir()):
        if not fold_dir.is_dir() or not fold_dir.name.startswith("Fold"):
            continue
        for subj_dir in sorted(fold_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            for video_file in sorted(subj_dir.iterdir()):
                if not video_file.is_file():
                    continue
                rel_path = str(video_file)
                state = get_state(video_file.name)
                rows.append([video_id, rel_path, state])
                video_id += 1

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "video_path", "state"])
        writer.writerows(rows)

    print(f"Generated {OUTPUT_CSV} with {len(rows)} entries")


if __name__ == "__main__":
    main()
