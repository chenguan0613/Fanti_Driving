from src.train import Train


def main():
    PATH = "./data/dataset/merge_five_enhanced.csv"
    t = Train(PATH)
    t.train_and_eval()


if __name__ == "__main__":
    main()
