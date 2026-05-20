import json
import time
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# wspólne cechy wygenerowane wcześniej przez svm_bird_audio.py
SHARED_FEATURES_ROOT = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\results_classical_shared_aug")
FEATURES_DIR = SHARED_FEATURES_ROOT / "features"

OUTPUT_DIR = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\results_classical_rf")
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_MAX_FEATURES = "sqrt"
RF_CLASS_WEIGHT = "balanced"

# dodatkowe techniczne
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF = 1
RF_RANDOM_STATE = 42
RF_N_JOBS = -1

FORCE_RETRAIN_MODEL = True

def ensure_dirs() -> None:
    for directory in [OUTPUT_DIR, PLOTS_DIR, MODELS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}")

def load_features(split_name: str) -> pd.DataFrame:
    path = FEATURES_DIR / f"{split_name}_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku cech: {path}\n"
            f"Najpierw uruchom svm_bird_audio.py, aby wygenerować wspólne cechy z augmentacją."
        )
    log(f"Wczytuję cechy: {path}")
    return pd.read_csv(path)


def get_feature_columns(df: pd.DataFrame):
    excluded = {"path", "label", "split", "augmentation"}
    return [col for col in df.columns if col not in excluded]


def prepare_datasets(train_df, val_df, test_df):
    feature_cols = get_feature_columns(train_df)

    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values

    y_train_raw = train_df["label"].values
    y_val_raw = val_df["label"].values
    y_test_raw = test_df["label"].values

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)
    y_test = label_encoder.transform(y_test_raw)

    return (
        feature_cols,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        label_encoder
    )

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def save_classification_report(y_true, y_pred, class_names, split_name):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    report_path = REPORTS_DIR / f"classification_report_{split_name}.csv"
    report_df.to_csv(report_path, index=True)
    log(f"Zapisano classification report dla {split_name}: {report_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, split_name):
    cm = confusion_matrix(y_true, y_pred)

    fig_width = max(14, len(class_names) * 0.22)
    fig_height = max(12, len(class_names) * 0.22)

    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Macierz pomyłek - Random Forest ({split_name})")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90, fontsize=7)
    plt.yticks(ticks, class_names, fontsize=7)
    plt.ylabel("Klasa rzeczywista")
    plt.xlabel("Klasa przewidziana")
    plt.tight_layout()

    plot_path = PLOTS_DIR / f"confusion_matrix_{split_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    csv_path = REPORTS_DIR / f"confusion_matrix_{split_name}.csv"
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(csv_path)

    log(f"Zapisano macierz pomyłek ({split_name}) do: {plot_path}")
    log(f"Zapisano wartości macierzy pomyłek ({split_name}) do: {csv_path}")


def save_feature_importance(model, feature_cols):
    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    csv_path = REPORTS_DIR / "feature_importance.csv"
    df.to_csv(csv_path, index=False)
    log(f"Zapisano ważność cech do: {csv_path}")

    top_n = min(20, len(df))
    top_df = df.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(top_df["feature"][::-1], top_df["importance"][::-1])
    plt.title("Najważniejsze cechy - Random Forest")
    plt.xlabel("Ważność cech")
    plt.tight_layout()

    plot_path = PLOTS_DIR / "feature_importance_top20.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    log(f"Zapisano wykres ważności cech do: {plot_path}")

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        max_features=RF_MAX_FEATURES,
        class_weight=RF_CLASS_WEIGHT,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RF_RANDOM_STATE,
        n_jobs=RF_N_JOBS,
    )

    log("Rozpoczynam trening Random Forest...")
    start = time.time()
    model.fit(X_train, y_train)
    train_time_s = time.time() - start
    log(f"Trening Random Forest zakończony. Czas treningu: {train_time_s:.2f} s")

    return model, train_time_s

def main():
    ensure_dirs()
    log("=== START PIPELINE RANDOM FOREST ===")

    # 1.Wczytanie wspólnych cech
    train_df = load_features("train")
    val_df = load_features("val")
    test_df = load_features("test")

    # 2.Przygotowanie danych
    (
        feature_cols,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        label_encoder
    ) = prepare_datasets(train_df, val_df, test_df)

    class_names = list(label_encoder.classes_)

    log(f"Liczba cech wejściowych: {len(feature_cols)}")
    log(f"Liczba klas: {len(class_names)}")
    log(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    pd.Series(feature_cols, name="feature_name").to_csv(
        REPORTS_DIR / "feature_columns.csv", index=False
    )
    pd.Series(class_names, name="class_name").to_csv(
        REPORTS_DIR / "class_names.csv", index=False
    )

    # 3.Trening
    model_path = MODELS_DIR / "random_forest_model.joblib"
    encoder_path = MODELS_DIR / "label_encoder.joblib"

    if model_path.exists() and not FORCE_RETRAIN_MODEL:
        log(f"Wczytuję istniejący model: {model_path}")
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        train_time_s = None
    else:
        model, train_time_s = train_random_forest(X_train, y_train)
        joblib.dump(model, model_path)
        joblib.dump(label_encoder, encoder_path)

        log(f"Zapisano model do: {model_path}")
        log(f"Zapisano label encoder do: {encoder_path}")

    # 4.Ewaluacja
    summary_rows = []
    pred_times = {}

    for split_name, X_split, y_split in [
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        log(f"Predykcja dla splitu: {split_name}")
        pred_start = time.time()
        y_pred = model.predict(X_split)
        pred_time_s = time.time() - pred_start
        pred_times[split_name] = pred_time_s

        log(f"Czas predykcji ({split_name}): {pred_time_s:.2f} s")

        metrics = compute_metrics(y_split, y_pred)
        metrics["split"] = split_name
        metrics["model"] = "RandomForest"
        metrics["n_samples"] = int(len(y_split))
        summary_rows.append(metrics)

        save_classification_report(y_split, y_pred, class_names, split_name)
        plot_confusion_matrix(y_split, y_pred, class_names, split_name)

    # 5.Podsumowanie metryk
    summary_df = pd.DataFrame(summary_rows)
    summary_path = REPORTS_DIR / "rf_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log(f"Zapisano podsumowanie metryk do: {summary_path}")

    # 6.Ważność cech
    save_feature_importance(model, feature_cols)

    # 7.Rozszerzone podsumowanie eksperymentu
    test_metrics = summary_df[summary_df["split"] == "test"].iloc[0].to_dict()
    val_metrics = summary_df[summary_df["split"] == "val"].iloc[0].to_dict()

    experiment_summary = pd.DataFrame([{
        "model": "RandomForest",
        "input_type": "aggregated_features",
        "n_classes": len(class_names),
        "feature_dim": len(feature_cols),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "train_time_s": train_time_s,
        "pred_val_time_s": pred_times.get("val"),
        "pred_test_time_s": pred_times.get("test"),
        "val_accuracy": val_metrics["accuracy"],
        "val_precision_macro": val_metrics["precision_macro"],
        "val_recall_macro": val_metrics["recall_macro"],
        "val_f1_macro": val_metrics["f1_macro"],
        "val_f1_weighted": val_metrics["f1_weighted"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_f1_weighted": test_metrics["f1_weighted"],
        "shared_features_used": True,
        "features_source_dir": str(FEATURES_DIR),
    }])

    experiment_summary_path = REPORTS_DIR / "rf_experiment_summary.csv"
    experiment_summary.to_csv(experiment_summary_path, index=False)
    log(f"Zapisano rozszerzone podsumowanie eksperymentu do: {experiment_summary_path}")

    # 8.Konfiguracja eksperymentu
    config = {
        "shared_features_root": str(SHARED_FEATURES_ROOT),
        "features_dir": str(FEATURES_DIR),
        "output_dir": str(OUTPUT_DIR),
        "rf_params": {
            "n_estimators": RF_N_ESTIMATORS,
            "max_depth": RF_MAX_DEPTH,
            "max_features": RF_MAX_FEATURES,
            "class_weight": RF_CLASS_WEIGHT,
            "min_samples_split": RF_MIN_SAMPLES_SPLIT,
            "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
            "random_state": RF_RANDOM_STATE,
            "n_jobs": RF_N_JOBS,
        },
        "timings": {
            "train_time_s": train_time_s,
            "pred_val_time_s": pred_times.get("val"),
            "pred_test_time_s": pred_times.get("test"),
        }
    }

    config_path = REPORTS_DIR / "experiment_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    log(f"Zapisano konfigurację eksperymentu do: {config_path}")
    log("=== KONIEC PIPELINE RANDOM FOREST ===")


if __name__ == "__main__":
    main()