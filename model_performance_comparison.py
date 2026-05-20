import pandas as pd
import matplotlib.pyplot as plt

paths = {
    "SVM": r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\results_classical_svm\reports\svm_experiment_summary.csv",
    "Random Forest": r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\results_classical_rf\reports\rf_experiment_summary.csv",
    "kNN": r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\results_classical_knn\reports\knn_experiment_summary.csv",
    "CNN": r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\results_deep_cnn\reports\cnn_experiment_summary.csv",
    "LSTM": r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\results_deep_lstm\reports\lstm_experiment_summary.csv",
}

dfs = []

for name, path in paths.items():
    df = pd.read_csv(path)
    df["model_name"] = name
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

table = df_all[[
    "model_name",
    "test_accuracy",
    "test_precision_macro",
    "test_recall_macro",
    "test_f1_macro",
    "test_f1_weighted"
]].copy()

table = table.sort_values(by="test_f1_macro", ascending=False)
time_table = df_all[[
    "model_name",
    "train_time_s",
    "pred_test_time_s"
]].copy()

print("\n=== CZASY ===")
print(time_table)

time_table.to_csv("tabela_czasow.csv", index=False)
print("\n=== TABELA WYNIKÓW ===")
print(table)

table.to_csv("tabela_wynikow.csv", index=False)

# WYKRES F1
plt.figure(figsize=(8,5))
plt.bar(table["model_name"], table["test_f1_macro"])
plt.xlabel("Model")
plt.ylabel("F1 macro")
plt.title("Porównanie modeli (F1 macro)")
plt.xticks(rotation=30)
plt.tight_layout()

plt.figure(figsize=(8,5))
plt.bar(time_table["model_name"], time_table["train_time_s"])
plt.xlabel("Model")
plt.ylabel("Czas treningu [s]")
plt.title("Porównanie czasu treningu modeli")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("wykres_czas_treningu.png", dpi=300)
plt.show()

plt.savefig("wykres_f1_macro.png", dpi=300)
plt.show()