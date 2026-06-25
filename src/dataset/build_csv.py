import csv
import zipfile
from pathlib import Path


class CSVBuilder:
    """
    CSVBuilder iterates videos under `DATA_DIR` recursively,
    and outputs all videos with labels.
    """

    DATA_DIR: Path
    OUTPUT_CSV: Path

    def __init__(
        self,
        data_dir: str = "data",
        output_csv: str = "src/dataset/dataset.csv",
    ) -> None:
        self.DATA_DIR = Path(data_dir)
        self.OUTPUT_CSV = Path(output_csv)

    def _get_state(self, filename: str) -> int:
        # file named "0" label it as "0"
        # file named "10" label it as "1"
        stem = Path(filename).stem
        if stem == "0":
            return 0
        elif stem == "10":
            return 1

    def _extract_zip_if_needed(self, zip_path: Path) -> Path | None:
        # Predict the folder path after decompression
        extract_dir = zip_path.with_suffix("")
        if extract_dir.exists():
            return extract_dir
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.DATA_DIR)
        return extract_dir if extract_dir.exists() else None

    def build_csv(self):
        rows = []
        video_id = 1

        fold_candidates = sorted(self.DATA_DIR.iterdir())
        # Traverse Fold levels
        for candidate in fold_candidates:
            fold_dir = None
            if candidate.is_dir() and candidate.name.startswith("Fold"):
                fold_dir = candidate
            elif candidate.suffix == ".zip" and candidate.stem.startswith("Fold"):
                fold_dir = self._extract_zip_if_needed(candidate)

            if fold_dir is None:
                continue
            # Traverse Subject levels
            for subj_dir in sorted(fold_dir.iterdir()):
                if not subj_dir.is_dir():
                    continue
                # Traverse the specific video files in the subject's folder
                for video_file in sorted(subj_dir.iterdir()):
                    if not video_file.is_file():
                        continue
                    # Exclude the unused video (named "5")
                    if video_file.stem == "5":
                        continue
                    rel_path = str(video_file)
                    state = self._get_state(video_file.name)
                    subject_id = f"{fold_dir.name}_{subj_dir.name}"
                    rows.append([video_id, rel_path, subject_id, state])
                    video_id += 1

        with open(self.OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "video_path", "subject_id", "state"])
            writer.writerows(rows)

        print(f"Generated {self.OUTPUT_CSV} with {len(rows)} entries")
