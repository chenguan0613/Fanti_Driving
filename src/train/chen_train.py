from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os
import joblib


class Train:
    def __init__(self, dataset_path: str) -> None:
        self.df = pd.read_csv(dataset_path)

        feature_cols = [
            c
            for c in self.df.columns
            if c
            not in (
                "video_id",
                "subject_id",
                "label",
                "yaw_mean",
                "yaw_std",
                "yaw_std_norm",
                # TODO: Check the effect of `pitch_mean`.
                "pitch_mean",
                "gaze_x_mean",
                "gaze_y_mean",
                "blink_rate",
                "blink_rate_norm",
            )
            and self.df[c].dtype in ("float64", "float32", "int64", "int32")
        ]

        self.features_names = list(feature_cols)
        X = self.df[feature_cols].values
        y = self.df["label"].values
        groups = self.df["subject_id"].values

        self.subjects = np.unique(groups)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

        self.X_train, self.X_test = X[train_idx], X[test_idx]
        self.y_train, self.y_test = y[train_idx], y[test_idx]
        self.train_subjects = np.unique(groups[train_idx])
        self.test_subjects = np.unique(groups[test_idx])
        print(f"Train subjects: {self.train_subjects}")
        print(f"Test subjects: {self.test_subjects}")
        print(f"Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")
        print(f"Train class ratio: {np.bincount(self.y_train.astype(int))}")
        print(f"Test class ratio: {np.bincount(self.y_test.astype(int))}")
        print()

    def train_and_eval(self):
        models = [
            ("Random Forest", self._random_forest()),
            ("SVM", self._svm()),
            ("Decision Tree", self._decision_tree()),
            ("Logistic Regression", self._logistic_regression()),
        ]

        best_score = -1
        best_model = None
        best_name = ""

        for name, model in models:
            print(f"--- {name} ---")
            y_pred = self._train(model)
            score = f1_score(self.y_test, y_pred, average="macro")
            self._eval(y_pred)
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
            print()
        os.makedirs("models", exist_ok=True)
        save_path = "models/fatigue_model.pkl"
        model_data = {"model": best_model, "feature_names": self.features_names}
        joblib.dump(model_data, save_path)
        print(f"\nBest model: {best_name} saved to {save_path}")

    def _train(self, model):
        model.fit(self.X_train, self.y_train)
        # if hasattr(model, "feature_importances_"):
        #     importances = model.feature_importances_
        #     if self.features_names is not None:
        #         for name, score in sorted(
        #             zip(self.features_names, importances), key=lambda x: -x[1]
        #         ):
        #             print(name, score)
        #     else:
        #         print(importances)
        y_pred = model.predict(self.X_test)
        return y_pred

    def _eval(self, y_pred):
        print(classification_report(self.y_test, y_pred, digits=3))

    def _random_forest(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=800,
            max_depth=25,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    def _svm(self) -> Pipeline:
        return make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
            ),
        )

    def _decision_tree(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            max_depth=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
        )

    def _logistic_regression(self) -> Pipeline:
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        )


def main():
    # path = "./src/dataset/training_features.csv"
    path = "./src/dataset/merge_five_enhanced.csv"
    t = Train(path)
    t.train_and_eval()


if __name__ == "__main__":
    main()
