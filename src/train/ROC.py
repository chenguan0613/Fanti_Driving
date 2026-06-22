import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from src.features import GOLDEN_FEATURES


class ModelEvaluator:
    def __init__(self, dataset_path: str):
        self.df = pd.read_csv(dataset_path)

    def run_evaluation(
        self,
        selected_features: list,
        output_image_path: str = "./data/visualization/roc_comparison.png",
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
                {"svc__estimator__C": [0.1, 1, 10]},
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

        plt.figure(figsize=(10, 8))
        print(f"[INFO] All models evaluation, dimension: {len(selected_features)}\n")
        for name, (model, model_params) in models_params.items():
            grid = GridSearchCV(
                model, model_params, cv=3, scoring="f1_macro", n_jobs=-1
            )
            grid.fit(X_train, y_train)
            best_curr = grid.best_estimator_
            y_pred = best_curr.predict(X_test)
            if hasattr(best_curr, "predict_proba"):
                y_prob = best_curr.predict_proba(X_test)[:, 1]
            else:
                y_prob = best_curr.decision_function(X_test)
            # F1-score
            f1 = f1_score(y_test, y_pred, average="macro")
            # TPR & FPR
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            single_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            single_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            # AUC & ROC array
            fpr_array, tpr_array, _ = roc_curve(y_test, y_prob)
            model_auc = auc(fpr_array, tpr_array)
            print("=" * 30)
            print(f"Model: {name}")
            print(f"--F1 score: {f1:.4f}--")
            print(f"--AUC area: {model_auc:.4f}--")
            print(
                f"--TPR (Recall): {single_tpr:.4f} | FPR(False alarm): {single_fpr:.4f}--"
            )
            plt.plot(
                fpr_array, tpr_array, lw=2, label=f"{name} (AUC = {model_auc:.4f})"
            )
        plt.plot(
            [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Chance"
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FPR)", fontsize=12)
        plt.ylabel("True Positive Rate (TPR)", fontsize=12)
        plt.title(
            "ROC Curves Comparison of Fatigue Detection Models",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
        print(f"\n[SUCCESS] ROC Compariation Map has saved into: {output_image_path}")
        plt.show()


if __name__ == "__main__":
    path = "./data/dataset/enhanced_merged.csv"
    evaluator = ModelEvaluator(path)
    evaluator.run_evaluation(GOLDEN_FEATURES)
