from src.train import Train
from src.train import HeuristicTrain


def main():
    PATH = "./data/dataset/enhanced_merged.csv"
    # train.py
    t = Train(PATH)
    t.train_and_eval("models/fatigue_model.pkl")

    print("=" * 20)
    print("After using heuristic learning to automatically select the features")
    print("=" * 20)

    # heuristic_train.py
    T = HeuristicTrain(PATH)
    T.run()


if __name__ == "__main__":
    main()
