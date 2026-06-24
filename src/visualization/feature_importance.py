import joblib
import pandas as pd
import matplotlib.pyplot as plt

data = joblib.load("models/heuristic_model.pkl")
model = data["model"]
features = data["feature_names"]
# Random Forest / Tree models
if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_
# Logistic Regression
elif hasattr(model, "coef_"):
    importance = model.coef_[0]
else:
    print("Feature importance not available for this model.")
    exit()
df = pd.DataFrame({"Feature": features, "Importance": importance})
df = df.sort_values(by="Importance", ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(df["Feature"], df["Importance"])
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("./data/visualization/feature_importance.png", dpi=300)
plt.show()
