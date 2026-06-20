import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


class Train:
    def __init__(self, dataset_path: str):
        self.df = pd.read_csv(dataset_path)

    def run(
        self, selected_features: list, output_path: str = "models/heuristic_model.pkl"
    ):
        X = self.df[selected_features].values
        y = self.df["label"].values
        groups = self.df["subject_id"].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        models_params = {
            "Random Forest": (
                RandomForestClassifier(
                    class_weight="balanced", random_state=42, n_jobs=-1
                ),
                {"n_estimators": [500, 800], "max_depth": [20, 25]},
            ),
            "SVM": (
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "svc",
                            CalibratedClassifierCV(
                                SVC(kernel="rbf", class_weight="balanced"),
                                method="sigmoid",
                            ),
                        ),
                    ]
                ),
                {"svc__estimator__C": [0.1, 1, 10]},  # 注意参数名前缀变化
            ),
            "Decision Tree": (
                DecisionTreeClassifier(class_weight="balanced", random_state=42),
                {"max_depth": [10, 20], "min_samples_leaf": [5, 10]},
            ),
            "Logistic Regression": (
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=42
                    ),
                ),
                {"logisticregression__C": [0.1, 1, 10]},
            ),
        }
        best_score = -1
        best_model = None
        best_name = ""
        for name, (model, model_params) in models_params.items():
            grid = GridSearchCV(
                model, model_params, cv=3, scoring="f1_macro", n_jobs=-1
            )
            grid.fit(X_train, y_train)

            best_curr = grid.best_estimator_
            y_pred = best_curr.predict(X_test)
            score = f1_score(y_test, y_pred, average="macro")
            print(f"--- {name:20} | F1-Score: {score:.4f} ---")
            print(classification_report(y_test, y_pred, digits=3))

            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        os.makedirs("models", exist_ok=True)
        joblib.dump(
            {"model": best_model, "feature_names": selected_features}, output_path
        )
        print(f"\nThe Best Model: {best_name} (F1: {best_score:.4f})")
        print(f"Model has been packeted into: {output_path}")


if __name__ == "__main__":
    GOLDEN_FEATURES = [
        "ear_std",
        "mar_mean",
        "mar_max",
        "pitch_std",
        "yaw_mean",
        "yaw_std",
        "gaze_y_mean",
        "ear_mean_norm",
        "mar_max_norm",
        "yaw_std_norm",
        "fatigue_index",
    ]
    path = "./src/dataset/features_enhanced.csv"
    arena = Train(path)
    arena.run(GOLDEN_FEATURES)
