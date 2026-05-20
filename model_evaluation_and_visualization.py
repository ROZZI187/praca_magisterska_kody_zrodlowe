import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BASE = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska")
OUT = BASE / "chapter7_outputs"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_CONFIG = {
    "SVM": {
        "summary": BASE / "results_classical_svm" / "reports" / "svm_experiment_summary.csv",
        "report": BASE / "results_classical_svm" / "reports" / "classification_report_test.csv",
        "cm_png": BASE / "results_classical_svm" / "plots" / "confusion_matrix_test.png",
    },
    "Random Forest": {
        "summary": BASE / "results_classical_rf" / "reports" / "rf_experiment_summary.csv",
        "report": BASE / "results_classical_rf" / "reports" / "classification_report_test.csv",
        "cm_png": BASE / "results_classical_rf" / "plots" / "confusion_matrix_test.png",
    },
    "kNN": {
        "summary": BASE / "results_classical_knn" / "reports" / "knn_experiment_summary.csv",
        "report": BASE / "results_classical_knn" / "reports" / "classification_report_test.csv",
        "cm_png": BASE / "results_classical_knn" / "plots" / "confusion_matrix_test.png",
    },
    "CNN": {
        "summary": BASE / "results_deep_cnn" / "reports" / "cnn_experiment_summary.csv",
        "report": BASE / "results_deep_cnn" / "reports" / "classification_report_test.csv",
        "cm_png": BASE / "results_deep_cnn" / "plots" / "confusion_matrix_test.png",
    },
    "LSTM": {
        "summary": BASE / "results_deep_lstm" / "reports" / "lstm_experiment_summary.csv",
        "report": BASE / "results_deep_lstm" / "reports" / "classification_report_test.csv",
        "cm_png": BASE / "results_deep_lstm" / "plots" / "confusion_matrix_test.png",
    },
}

summary_rows = []
per_class_f1 = []

for model_name, paths in MODEL_CONFIG.items():
    df_sum = pd.read_csv(paths["summary"])
    row = df_sum.iloc[0].to_dict()
    row["model_name"] = model_name
    summary_rows.append(row)

    df_rep = pd.read_csv(paths["report"], index_col=0)
    df_rep = df_rep.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
    if "f1-score" in df_rep.columns:
        f1_series = df_rep["f1-score"].copy()
        f1_series.name = model_name
        per_class_f1.append(f1_series)

summary_df = pd.DataFrame(summary_rows)
order = ["CNN", "SVM", "LSTM", "Random Forest", "kNN"]
summary_df["model_name"] = pd.Categorical(summary_df["model_name"], categories=order, ordered=True)
summary_df = summary_df.sort_values("model_name")

metrics_table = summary_df[
    [
        "model_name",
        "test_accuracy",
        "test_precision_macro",
        "test_recall_macro",
        "test_f1_macro",
        "test_f1_weighted",
    ]
].copy()

metrics_table.columns = [
    "Model",
    "Accuracy",
    "Precision_macro",
    "Recall_macro",
    "F1_macro",
    "F1_weighted",
]

metrics_table.to_csv(OUT / "tab_7_2_metrics.csv", index=False)

metrics_table_round = metrics_table.copy()
for c in ["Accuracy", "Precision_macro", "Recall_macro", "F1_macro", "F1_weighted"]:
    metrics_table_round[c] = metrics_table_round[c].round(4)
metrics_table_round.to_csv(OUT / "tab_7_2_metrics_rounded.csv", index=False)
time_table = summary_df[
    [
        "model_name",
        "train_time_s",
        "pred_test_time_s",
    ]
].copy()

time_table.columns = [
    "Model",
    "Train_time_s",
    "Prediction_time_test_s",
]

time_table_round = time_table.copy()
time_table_round["Train_time_s"] = time_table_round["Train_time_s"].round(2)
time_table_round["Prediction_time_test_s"] = time_table_round["Prediction_time_test_s"].round(2)

time_table_round.to_csv(OUT / "tab_7_3_times.csv", index=False)

#Accuracy
plt.figure(figsize=(9, 5))
plt.bar(metrics_table["Model"], metrics_table["Accuracy"])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Porównanie modeli na podstawie accuracy")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(OUT / "fig_accuracy.png", dpi=300, bbox_inches="tight")
plt.close()

#F1 macro
plt.figure(figsize=(9, 5))
plt.bar(metrics_table["Model"], metrics_table["F1_macro"])
plt.xlabel("Model")
plt.ylabel("F1 macro")
plt.title("Porównanie modeli na podstawie F1 macro")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(OUT / "fig_f1_macro.png", dpi=300, bbox_inches="tight")
plt.close()

#F1 weighted
plt.figure(figsize=(9, 5))
plt.bar(metrics_table["Model"], metrics_table["F1_weighted"])
plt.xlabel("Model")
plt.ylabel("F1 weighted")
plt.title("Porównanie modeli na podstawie F1 weighted")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(OUT / "fig_f1_weighted.png", dpi=300, bbox_inches="tight")
plt.close()

#Precision / Recall / F1 macro
plot_df = metrics_table.copy()
x = np.arange(len(plot_df))
width = 0.24

plt.figure(figsize=(10, 5))
plt.bar(x - width, plot_df["Precision_macro"], width, label="Precision macro")
plt.bar(x, plot_df["Recall_macro"], width, label="Recall macro")
plt.bar(x + width, plot_df["F1_macro"], width, label="F1 macro")

plt.xlabel("Model")
plt.ylabel("Wartość metryki")
plt.title("Porównanie precision, recall i F1 macro")
plt.xticks(x, plot_df["Model"], rotation=25)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "fig_precision_recall_f1_macro.png", dpi=300, bbox_inches="tight")
plt.close()

#heatmapa F1 per class
heatmap_df = pd.concat(per_class_f1, axis=1)
heatmap_df = heatmap_df[order]
heatmap_df.to_csv(OUT / "per_class_f1_all_models.csv")

plt.figure(figsize=(10, 14))
plt.imshow(heatmap_df.fillna(0).values, aspect="auto")
plt.colorbar(label="F1-score")
plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=25)
plt.yticks(range(len(heatmap_df.index)), heatmap_df.index, fontsize=7)
plt.xlabel("Model")
plt.ylabel("Klasa")
plt.title("Porównanie wartości F1-score dla poszczególnych klas")
plt.tight_layout()
plt.savefig(OUT / "fig_heatmap_per_class_f1.png", dpi=300, bbox_inches="tight")
plt.close()

#zbiorczy rysunek z macierzami pomyłek
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for ax, model_name in zip(axes, order):
    img = mpimg.imread(MODEL_CONFIG[model_name]["cm_png"])
    ax.imshow(img)
    ax.set_title(model_name)
    ax.axis("off")

if len(order) < len(axes):
    for i in range(len(order), len(axes)):
        axes[i].axis("off")

plt.tight_layout()
plt.savefig(OUT / "fig_confusion_matrix_grid.png", dpi=300, bbox_inches="tight")
plt.close()

print("Gotowe. Pliki zapisano w folderze:")
print(OUT)