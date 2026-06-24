import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

results_fs = {
    "Model": ["Random Forest", "SVM", "Decision Tree", "Logistic Regression"],
    "Accuracy": [0.842, 0.824, 0.814, 0.769],
    "F1": [0.8424, 0.8238, 0.8131, 0.7653],
    "Precision": [0.844, 0.831, 0.814, 0.801],
    "Recall": [0.842, 0.824, 0.813, 0.769],
}
results_sys = {
    "Model": ["Random Forest", "SVM", "Decision Tree", "Logistic Regression"],
    "Accuracy": [0.877, 0.852, 0.850, 0.834],
    "F1": [0.8765, 0.8518, 0.8500, 0.8322],
    "Precision": [0.878, 0.860, 0.852, 0.854],
    "Recall": [0.877, 0.852, 0.850, 0.834],
}
df_fs = pd.DataFrame(results_fs)
df_sys = pd.DataFrame(results_sys)
x = np.arange(len(df_fs["Model"]))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(x - width / 2, df_fs["Accuracy"], width, label="Unfiltered")
plt.bar(x + width / 2, df_sys["Accuracy"], width, label="Filtered")
plt.xticks(x, df_fs["Model"], rotation=20)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison (Unfiltered vs Filterd)")
plt.legend()
plt.ylim(0.60, 1.00)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("./data/visualization/accuracy_comparison_fs_vs_sys.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(x - width / 2, df_fs["F1"], width, label="Unfiltered")
plt.bar(x + width / 2, df_sys["F1"], width, label="Filtered")
plt.xticks(x, df_fs["Model"], rotation=20)
plt.ylabel("F1 Score")
plt.title("Model F1 Score Comparison (Unfiltered vs Filterd))")
plt.legend()
plt.ylim(0.60, 1.00)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("./data/visualization/f1_comparison_fs_vs_sys.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(x - width / 2, df_fs["Precision"], width, label="Unfiltered")
plt.bar(x + width / 2, df_sys["Precision"], width, label="Filtered")
plt.xticks(x, df_fs["Model"], rotation=20)
plt.ylabel("Precision")
plt.title("Model Precision Comparison (Unfiltered vs Filterd)")
plt.legend()
plt.ylim(0.60, 1.00)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("./data/visualization/precision_comparison_fs_vs_sys.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(x - width / 2, df_fs["Recall"], width, label="Unfiltered")
plt.bar(x + width / 2, df_sys["Recall"], width, label="Filtered")
plt.xticks(x, df_fs["Model"], rotation=20)
plt.ylabel("Recall")
plt.title("Model Recall Comparison (Unfiltered vs Filterd))")
plt.legend()
plt.ylim(0.60, 1.00)
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("./data/visualization/recall_comparison_fs_vs_sys.png", dpi=300)
plt.close()
