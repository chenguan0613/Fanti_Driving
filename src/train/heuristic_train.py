import pandas as pd
import os
import joblib
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, f1_score
from src.features import GOLDEN_FEATURES
from .model_config import MODEL_PARAMS


class HeuristicTrain:
    def __init__(self, dataset_path: str):
        self.df = pd.read_csv(dataset_path)

    def run(self, output_path: str = "models/heuristic_model.pkl"):
        X = self.df[GOLDEN_FEATURES].values
        y = self.df["label"].values
        groups = self.df["subject_id"].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        models_params = MODEL_PARAMS
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
                best_model = best_curr
                best_name = name
        os.makedirs("models", exist_ok=True)
        joblib.dump(
            {"model": best_model, "feature_names": GOLDEN_FEATURES}, output_path
        )
        print(f"\nThe Best Model: {best_name} (F1: {best_score:.4f})")
        print(f"Model has been packeted into: {output_path}")
