from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

MODEL_PARAMS = {
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
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
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
        {"logisticregression__C": [0.1, 1, 10]},
    ),
}
