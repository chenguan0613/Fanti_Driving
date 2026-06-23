from src.train import HeuristicTrain


def main():
    PATH = "./data/dataset/enhanced_merged.csv"
    # Using the dataset to train
    T = HeuristicTrain(PATH)
    T.run("models/heuristic_model.pkl")


if __name__ == "__main__":
    main()
