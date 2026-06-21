from multiprocessing import cpu_count
from src.dataset import FeatureExtractor, FeatureBaseline


def main():
    # `DATASET_LIST_PATH` can be obtained from `build_csv.py`
    DATASET_LIST_PATH = "src/dataset/dataset.csv"
    RAW_FEATURE_PATH = "./training_features.csv"
    ENHANCED_FEATURE_PATH = "./enhanced.csv"
    WORKDERS = cpu_count()

    fe = FeatureExtractor(30)
    fe.extract(
        dataset_list_path=DATASET_LIST_PATH,
        output_path=RAW_FEATURE_PATH,
        workers=WORKDERS,
    )

    fb = FeatureBaseline(RAW_FEATURE_PATH)
    fb.process_and_save(ENHANCED_FEATURE_PATH)


if __name__ == "__main__":
    main()
