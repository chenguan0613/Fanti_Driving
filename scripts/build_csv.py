from src.dataset.build_csv import CSVBuilder


def main():
    DATA_DIR = "data/"
    OUTPUT_PATH = "data/dataset.csv"

    cb = CSVBuilder(DATA_DIR, OUTPUT_PATH)
    cb.build_csv()


if __name__ == "__main__":
    main()
