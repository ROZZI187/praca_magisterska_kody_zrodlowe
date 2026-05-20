import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska")

models = {
    "CNN": BASE / r"results_deep_cnn\reports\confusion_matrix_test.csv",
    "LSTM": BASE / r"results_deep_lstm\reports\confusion_matrix_test.csv",
    "SVM": BASE / r"results_classical_svm\reports\confusion_matrix_test.csv",
    "Random Forest": BASE / r"results_classical_rf\reports\confusion_matrix_test.csv",
    "kNN": BASE / r"results_classical_knn\reports\confusion_matrix_test.csv",
}

out_rows = []

for model_name, path in models.items():
    cm_df = pd.read_csv(path, index_col=0)
    labels = cm_df.index.tolist()
    cm = cm_df.values.copy()

    np.fill_diagonal(cm, 0)

    flat_idx = np.argsort(cm.ravel())[::-1][:10]

    for idx in flat_idx:
        true_i, pred_j = np.unravel_index(idx, cm.shape)
        count = int(cm[true_i, pred_j])

        if count == 0:
            continue

        out_rows.append({
            "model": model_name,
            "klasa_rzeczywista": labels[true_i],
            "klasa_przewidziana": labels[pred_j],
            "liczba_pomyłek": count
        })

result = pd.DataFrame(out_rows)
save_path = BASE / r"results_chapter7_common\top_confusions_test.csv"
save_path.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(save_path, index=False, encoding="utf-8-sig")

print(result)
print(f"\nZapisano do: {save_path}")