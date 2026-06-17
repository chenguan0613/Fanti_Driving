from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class Train:
    def __init__(self, dataset_path: str) -> None:
        self.df = pd.read_csv(dataset_path)
        X = self.df.drop(columns=["label"]).values
        y = self.df["label"].values
        groups = self.df["video_id"].values

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

        self.X_train, self.X_test = X[train_idx], X[test_idx]
        self.y_train, self.y_test = y[train_idx], y[test_idx]

    def train_and_eval(self):
        models = [
            self._random_forest(),
            self._svm(),
            self._decision_tree(),
            self._logistic_regression(),
        ]

        for model in models:
            y_pred = self._train(model)
            self._eval(y_pred)

    def _train(
        self,
        model: RandomForestClassifier | Pipeline | DecisionTreeClassifier,
    ):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return y_pred

    def _eval(self, y_pred):
        print(classification_report(self.y_test, y_pred))

    def _random_forest(self) -> RandomForestClassifier:
        rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
        return rf

    def _svm(self) -> Pipeline:
        svm = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
            ),
        )
        return svm

    def _decision_tree(self) -> DecisionTreeClassifier:
        dt = DecisionTreeClassifier(max_depth=10, random_state=42)
        return dt

    def _logistic_regression(self) -> Pipeline:
        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )
        return lr


def main():
    path = "./src/dataset/training_features.csv"
    t = Train(path)
    t.train_and_eval()


if __name__ == "__main__":
    main()
