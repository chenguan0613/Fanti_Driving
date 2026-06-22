from src.train import HeuristicTrain


def main():
    PATH = "./data/dataset/enhanced_merged.csv"

    print("=" * 20)
    print("After using heuristic learning to automatically select the features")
    print("=" * 20)

    # heuristic_train.py
    T = HeuristicTrain(PATH)
    T.run("models/heuristic_model.pkl")


if __name__ == "__main__":
    main()
