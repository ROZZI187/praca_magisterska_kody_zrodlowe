import json
import time
import joblib
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\data_splits")
SHARED_FEATURES_ROOT = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\results_classical_shared_aug")
FEATURES_DIR = SHARED_FEATURES_ROOT / "features"
FEATURE_REPORTS_DIR = SHARED_FEATURES_ROOT / "reports"

OUTPUT_DIR = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\results_classical_svm")
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

# audio / cechy
SR = 22050
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
ROLLOFF_PERCENT = 0.85

USE_TRAIN_AUGMENTATION = True
TRAIN_AUGMENTATIONS = ["noise", "pitch_shift", "time_stretch", "time_shift"]

NOISE_SIGMA = 0.005
PITCH_SHIFT_STEPS = 1
TIME_STRETCH_RATE = 0.90
TIME_SHIFT_RATIO = 0.10

SVM_KERNEL = "rbf"
SVM_C = 10.0
SVM_GAMMA = "scale"
SVM_CLASS_WEIGHT = "balanced"
SVM_PROBABILITY = False
SVM_RANDOM_STATE = 42

# sterowanie
FORCE_REEXTRACT_FEATURES = True
FORCE_RETRAIN_MODEL = True

def ensure_dirs() -> None:
    for directory in [
        SHARED_FEATURES_ROOT, FEATURES_DIR, FEATURE_REPORTS_DIR,
        OUTPUT_DIR, PLOTS_DIR, MODELS_DIR, REPORTS_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)

def log(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


def safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if len(x) > 0 else 0.0


def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) > 0 else 0.0


def fix_length(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) > target_len:
        return y[:target_len]
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)), mode="constant")
    return y

def collect_split_files(split_dir: Path) -> List[Tuple[str, str]]:
    records = []

    if not split_dir.exists():
        raise FileNotFoundError(f"Nie znaleziono folderu split: {split_dir}")

    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

    for class_dir in class_dirs:
        label = class_dir.name
        audio_files = sorted([
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in [".wav", ".flac", ".mp3", ".ogg", ".m4a"]
        ])
        for audio_path in audio_files:
            records.append((str(audio_path), label))

    return records

# AUGMENTACJA
def augment_audio(y: np.ndarray, sr: int, aug_name: str) -> np.ndarray:
    target_len = len(y)

    if aug_name == "noise":
        noise = np.random.normal(0, NOISE_SIGMA, size=len(y))
        y_aug = np.clip(y + noise, -1.0, 1.0)
        return fix_length(y_aug.astype(np.float32), target_len)

    if aug_name == "pitch_shift":
        y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=PITCH_SHIFT_STEPS)
        return fix_length(y_aug.astype(np.float32), target_len)

    if aug_name == "time_stretch":
        y_aug = librosa.effects.time_stretch(y, rate=TIME_STRETCH_RATE)
        return fix_length(y_aug.astype(np.float32), target_len)

    if aug_name == "time_shift":
        shift = int(TIME_SHIFT_RATIO * target_len)
        if shift > 0:
            y_aug = np.concatenate([np.zeros(shift, dtype=y.dtype), y[:-shift]])
        else:
            y_aug = y.copy()
        return fix_length(y_aug.astype(np.float32), target_len)

    raise ValueError(f"Nieznana augmentacja: {aug_name}")

def extract_features_from_signal(y: np.ndarray, sr: int) -> Dict[str, float]:
    if y is None or len(y) == 0:
        raise ValueError("Pusty sygnał")

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH
    )

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, roll_percent=ROLLOFF_PERCENT, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y, hop_length=HOP_LENGTH
    )
    rms = librosa.feature.rms(
        y=y, frame_length=N_FFT, hop_length=HOP_LENGTH
    )

    features = {}

    for i in range(N_MFCC):
        coeff = mfcc[i]
        features[f"mfcc_{i+1}_mean"] = safe_mean(coeff)
        features[f"mfcc_{i+1}_std"] = safe_std(coeff)

    features["spectral_centroid_mean"] = safe_mean(spectral_centroid[0])
    features["spectral_centroid_std"] = safe_std(spectral_centroid[0])

    features["spectral_bandwidth_mean"] = safe_mean(spectral_bandwidth[0])
    features["spectral_bandwidth_std"] = safe_std(spectral_bandwidth[0])

    features["spectral_rolloff_mean"] = safe_mean(spectral_rolloff[0])
    features["spectral_rolloff_std"] = safe_std(spectral_rolloff[0])

    features["zcr_mean"] = safe_mean(zero_crossing_rate[0])
    features["zcr_std"] = safe_std(zero_crossing_rate[0])

    features["rms_mean"] = safe_mean(rms[0])
    features["rms_std"] = safe_std(rms[0])

    return features


def extract_features_for_split(split_name: str) -> Tuple[pd.DataFrame, Dict]:
    csv_path = FEATURES_DIR / f"{split_name}_features.csv"

    if csv_path.exists() and not FORCE_REEXTRACT_FEATURES:
        log(f"Wczytuję gotowe cechy: {csv_path}")
        df = pd.read_csv(csv_path)
        timing_info = {
            "split": split_name,
            "raw_audio_files": None,
            "feature_rows_written": int(len(df)),
            "extraction_time_s": None,
            "augmentation_enabled": split_name == "train" and USE_TRAIN_AUGMENTATION,
        }
        return df, timing_info

    records = collect_split_files(BASE_DIR / split_name)
    total = len(records)

    if total == 0:
        raise ValueError(f"Nie znaleziono plików audio dla splitu {split_name}")

    log(f"Split '{split_name}': znaleziono {total} plików audio")

    rows = []
    errors = []
    aug_counter = {
        "original": 0,
        "noise": 0,
        "pitch_shift": 0,
        "time_stretch": 0,
        "time_shift": 0,
    }

    split_start = time.time()

    for idx, (audio_path, label) in enumerate(records, start=1):
        try:
            y, sr = librosa.load(audio_path, sr=SR, mono=True)

            # oryginał
            feats = extract_features_from_signal(y, sr)
            row = {
                "path": audio_path,
                "label": label,
                "split": split_name,
                "augmentation": "original",
            }
            row.update(feats)
            rows.append(row)
            aug_counter["original"] += 1

            # augmentacja dla train
            if split_name == "train" and USE_TRAIN_AUGMENTATION:
                for aug_name in TRAIN_AUGMENTATIONS:
                    y_aug = augment_audio(y, sr, aug_name)
                    feats_aug = extract_features_from_signal(y_aug, sr)

                    row_aug = {
                        "path": audio_path,
                        "label": label,
                        "split": split_name,
                        "augmentation": aug_name,
                    }
                    row_aug.update(feats_aug)
                    rows.append(row_aug)
                    aug_counter[aug_name] += 1

        except Exception as e:
            errors.append({
                "path": audio_path,
                "label": label,
                "error": str(e)
            })

        if idx % 250 == 0 or idx == total:
            elapsed = time.time() - split_start
            log(f"[{split_name}] {idx}/{total} plików przetworzonych | czas: {elapsed:.1f} s")

    extraction_time = time.time() - split_start

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    log(f"Zapisano cechy splitu '{split_name}' do: {csv_path}")
    log(f"Liczba rekordów cech: {len(df)}")

    aug_summary_path = FEATURE_REPORTS_DIR / f"{split_name}_augmentation_summary.json"
    with open(aug_summary_path, "w", encoding="utf-8") as f:
        json.dump(aug_counter, f, indent=4, ensure_ascii=False)

    if errors:
        errors_df = pd.DataFrame(errors)
        errors_path = FEATURE_REPORTS_DIR / f"{split_name}_feature_errors.csv"
        errors_df.to_csv(errors_path, index=False)
        log(f"Błędy ekstrakcji ({len(errors)}) zapisano do: {errors_path}")

    timing_info = {
        "split": split_name,
        "raw_audio_files": int(total),
        "feature_rows_written": int(len(df)),
        "extraction_time_s": float(extraction_time),
        "augmentation_enabled": split_name == "train" and USE_TRAIN_AUGMENTATION,
        "augmentation_summary": aug_counter,
    }

    return df, timing_info


def load_all_features():
    train_df, train_time = extract_features_for_split("train")
    val_df, val_time = extract_features_for_split("val")
    test_df, test_time = extract_features_for_split("test")
    return train_df, val_df, test_df, train_time, val_time, test_time

def get_feature_columns(df: pd.DataFrame) -> List[str]:
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        feature_cols,
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        X_test_scaled, y_test,
        scaler, label_encoder
    )

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
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


def save_classification_report(y_true, y_pred, class_names: List[str], split_name: str) -> None:
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


def plot_confusion_matrix(y_true, y_pred, class_names: List[str], split_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)

    fig_width = max(14, len(class_names) * 0.22)
    fig_height = max(12, len(class_names) * 0.22)

    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Macierz pomyłek - SVM ({split_name})")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=7)
    plt.yticks(tick_marks, class_names, fontsize=7)
    plt.ylabel("Klasa rzeczywista")
    plt.xlabel("Klasa przewidziana")
    plt.tight_layout()

    plot_path = PLOTS_DIR / f"confusion_matrix_{split_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    cm_path = REPORTS_DIR / f"confusion_matrix_{split_name}.csv"
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(cm_path)

    log(f"Zapisano macierz pomyłek ({split_name}) do: {plot_path}")
    log(f"Zapisano wartości macierzy pomyłek ({split_name}) do: {cm_path}")

def train_svm(X_train, y_train):
    model = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        class_weight=SVM_CLASS_WEIGHT,
        probability=SVM_PROBABILITY,
        random_state=SVM_RANDOM_STATE
    )

    log("Rozpoczynam trening SVM...")
    start = time.time()
    model.fit(X_train, y_train)
    train_time_s = time.time() - start
    log(f"Trening SVM zakończony. Czas treningu: {train_time_s:.2f} s")

    return model, train_time_s

def main():
    ensure_dirs()
    log("=== START PIPELINE SVM Z AUGMENTACJĄ TRAIN ===")

    # 1.cechy i czasy ekstrakcji
    train_df, val_df, test_df, train_feat_time, val_feat_time, test_feat_time = load_all_features()

    # 2.przygotowanie danych
    (
        feature_cols,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        scaler, label_encoder
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

    # 3.trening
    model_path = MODELS_DIR / "svm_rbf_model.joblib"
    scaler_path = MODELS_DIR / "standard_scaler.joblib"
    encoder_path = MODELS_DIR / "label_encoder.joblib"

    if model_path.exists() and not FORCE_RETRAIN_MODEL:
        log(f"Wczytuję istniejący model: {model_path}")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)
        train_time_s = None
    else:
        model, train_time_s = train_svm(X_train, y_train)
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(label_encoder, encoder_path)

        log(f"Zapisano model do: {model_path}")
        log(f"Zapisano scaler do: {scaler_path}")
        log(f"Zapisano label encoder do: {encoder_path}")

    # 4.ewaluacja
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
        metrics["model"] = "SVM_RBF"
        metrics["n_samples"] = int(len(y_split))
        summary_rows.append(metrics)

        save_classification_report(y_split, y_pred, class_names, split_name)
        plot_confusion_matrix(y_split, y_pred, class_names, split_name)

    # 5.klasyczne podsumowanie metryk
    summary_df = pd.DataFrame(summary_rows)
    summary_path = REPORTS_DIR / "svm_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log(f"Zapisano podsumowanie metryk do: {summary_path}")

    # 6.rozszerzone podsumowanie eksperymentu
    total_feature_extraction_s = sum([
        t["extraction_time_s"] for t in [train_feat_time, val_feat_time, test_feat_time]
        if t["extraction_time_s"] is not None
    ])

    test_metrics = summary_df[summary_df["split"] == "test"].iloc[0].to_dict()
    val_metrics = summary_df[summary_df["split"] == "val"].iloc[0].to_dict()

    experiment_summary = pd.DataFrame([{
        "model": "SVM_RBF",
        "input_type": "aggregated_features",
        "n_classes": len(class_names),
        "feature_dim": len(feature_cols),
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "train_raw_audio_files": train_feat_time["raw_audio_files"],
        "val_raw_audio_files": val_feat_time["raw_audio_files"],
        "test_raw_audio_files": test_feat_time["raw_audio_files"],
        "train_feature_rows": train_feat_time["feature_rows_written"],
        "val_feature_rows": val_feat_time["feature_rows_written"],
        "test_feature_rows": test_feat_time["feature_rows_written"],
        "feature_extraction_train_s": train_feat_time["extraction_time_s"],
        "feature_extraction_val_s": val_feat_time["extraction_time_s"],
        "feature_extraction_test_s": test_feat_time["extraction_time_s"],
        "feature_extraction_total_s": total_feature_extraction_s,
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
        "train_augmentation_enabled": USE_TRAIN_AUGMENTATION,
        "train_augmentations": ",".join(TRAIN_AUGMENTATIONS) if USE_TRAIN_AUGMENTATION else "",
    }])

    experiment_summary_path = REPORTS_DIR / "svm_experiment_summary.csv"
    experiment_summary.to_csv(experiment_summary_path, index=False)
    log(f"Zapisano rozszerzone podsumowanie eksperymentu do: {experiment_summary_path}")

    config = {
        "base_dir": str(BASE_DIR),
        "shared_features_root": str(SHARED_FEATURES_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "audio_params": {
            "sr": SR,
            "n_mfcc": N_MFCC,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "win_length": WIN_LENGTH,
            "rolloff_percent": ROLLOFF_PERCENT,
        },
        "train_augmentation": {
            "enabled": USE_TRAIN_AUGMENTATION,
            "augmentations": TRAIN_AUGMENTATIONS,
            "noise_sigma": NOISE_SIGMA,
            "pitch_shift_steps": PITCH_SHIFT_STEPS,
            "time_stretch_rate": TIME_STRETCH_RATE,
            "time_shift_ratio": TIME_SHIFT_RATIO,
        },
        "svm_params": {
            "kernel": SVM_KERNEL,
            "C": SVM_C,
            "gamma": SVM_GAMMA,
            "class_weight": SVM_CLASS_WEIGHT,
            "probability": SVM_PROBABILITY,
            "random_state": SVM_RANDOM_STATE,
        },
        "timings": {
            "feature_extraction_train_s": train_feat_time["extraction_time_s"],
            "feature_extraction_val_s": val_feat_time["extraction_time_s"],
            "feature_extraction_test_s": test_feat_time["extraction_time_s"],
            "feature_extraction_total_s": total_feature_extraction_s,
            "train_time_s": train_time_s,
            "pred_val_time_s": pred_times.get("val"),
            "pred_test_time_s": pred_times.get("test"),
        }
    }

    config_path = REPORTS_DIR / "experiment_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    log(f"Zapisano konfigurację eksperymentu do: {config_path}")
    log("=== KONIEC PIPELINE SVM ===")


if __name__ == "__main__":
    main()