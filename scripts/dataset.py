from multiprocessing import cpu_count
from src.dataset import FeatureExtractor, FeatureBaseline


def main():
    # `DATASET_LIST_PATH` can be obtained from `build_csv.py`
    DATASET_LIST_PATH = "./data/dataset/dataset.csv"
    RAW_FEATURE_PATH = "./data/dataset/training_features.csv"
    ENHANCED_FEATURE_PATH = "./data/dataset/enhanced.csv"
    # WORKDERS = cpu_count()
    WORKDERS = 24

    fe = FeatureExtractor(
        window_stride=75,
        target_fps=30,
        window_size=150,
    )
    fe.extract(
        dataset_list_path=DATASET_LIST_PATH,
        output_path=RAW_FEATURE_PATH,
        workers=WORKDERS,
    )

    fb = FeatureBaseline(RAW_FEATURE_PATH)
    fb.process_and_save(ENHANCED_FEATURE_PATH)


if __name__ == "__main__":
    main()
