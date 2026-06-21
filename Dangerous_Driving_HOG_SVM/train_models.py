from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset_new"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def build_hog_descriptor(image_size):
    return cv2.HOGDescriptor(
        _winSize=image_size,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )


def extract_hog_features(image_path, image_size):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    resized = cv2.resize(image, image_size)
    resized = cv2.equalizeHist(resized)
    hog = build_hog_descriptor(image_size)
    return hog.compute(resized).reshape(-1)


def load_dataset(split, class_names, image_size):
    features = []
    labels = []

    for label in class_names:
        class_dir = DATASET_DIR / split / label
        for image_path in sorted(class_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            features.append(extract_hog_features(image_path, image_size))
            labels.append(label)

    return np.array(features), np.array(labels)


def train_classifier(name, class_names, image_size):
    print(f"\nTraining {name} model...")
    x_train, y_train = load_dataset("train", class_names, image_size)
    x_test, y_test = load_dataset("test", class_names, image_size)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(C=1.0, max_iter=10000, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=class_names)

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "class_names": class_names,
            "image_size": image_size,
        },
        MODELS_DIR / f"{name}_model.pkl",
    )

    report_path = REPORTS_DIR / f"{name}_classification_report.txt"
    matrix_path = REPORTS_DIR / f"{name}_confusion_matrix.txt"

    report_path.write_text(report, encoding="utf-8")
    matrix_path.write_text(
        "Labels: " + ", ".join(class_names) + "\n" + str(matrix),
        encoding="utf-8",
    )

    print(report)
    print("Confusion matrix:")
    print(matrix)


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError("dataset_new folder was not found.")

    train_classifier(
        name="eye",
        class_names=["Closed", "Open"],
        image_size=(64, 64),
    )
    train_classifier(
        name="yawn",
        class_names=["no_yawn", "yawn"],
        image_size=(96, 96),
    )

    print("\nDone. Models saved in the models folder.")


if __name__ == "__main__":
    main()
