import json
import math
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\data_splits")
OUTPUT_DIR = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\results_deep_cnn")
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

#audio
SR = 22050
DURATION = 5.0
TARGET_SAMPLES = int(SR * DURATION)

#Mel-spektrogram
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 0
FMAX = SR // 2

#augmentacja
USE_TRAIN_AUGMENTATION = True
TRAIN_AUGMENTATIONS = ["original", "noise", "pitch_shift", "time_stretch", "time_shift"]

NOISE_SIGMA = 0.005
PITCH_SHIFT_STEPS = 1
TIME_STRETCH_RATE = 0.90
TIME_SHIFT_RATIO = 0.10

CNN_FILTERS = [32, 64]
CNN_KERNEL_SIZE = (3, 3)
CNN_POOL_SIZE = (2, 2)
CNN_DROPOUT = 0.3
CNN_DENSE_UNITS = 128

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 20
RANDOM_STATE = 42

FORCE_REBUILD_SPLITS = True

#reproducibility
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def ensure_dirs() -> None:
    for directory in [OUTPUT_DIR, PLOTS_DIR, MODELS_DIR, REPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


def fix_length(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) > target_len:
        return y[:target_len]
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)), mode="constant")
    return y


def collect_split_files(split_dir: Path):
    records = []
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


def build_split_dataframe(split_name: str) -> pd.DataFrame:
    split_dir = BASE_DIR / split_name
    records = collect_split_files(split_dir)
    if not records:
        raise ValueError(f"Nie znaleziono plików dla splitu: {split_name}")

    df = pd.DataFrame(records, columns=["path", "label"])
    df["split"] = split_name
    return df


def save_split_dataframe(df: pd.DataFrame, split_name: str):
    out = REPORTS_DIR / f"{split_name}_file_index.csv"
    df.to_csv(out, index=False)
    log(f"Zapisano indeks plików: {out}")



# AUDIO / AUGMENTACJA / MEL
def load_audio(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    y = fix_length(y, TARGET_SAMPLES)
    return y.astype(np.float32)


def augment_audio(y: np.ndarray, aug_name: str) -> np.ndarray:
    if aug_name == "original":
        return y

    if aug_name == "noise":
        noise = np.random.normal(0, NOISE_SIGMA, size=len(y))
        y_aug = np.clip(y + noise, -1.0, 1.0)
        return fix_length(y_aug.astype(np.float32), TARGET_SAMPLES)

    if aug_name == "pitch_shift":
        y_aug = librosa.effects.pitch_shift(y, sr=SR, n_steps=PITCH_SHIFT_STEPS)
        return fix_length(y_aug.astype(np.float32), TARGET_SAMPLES)

    if aug_name == "time_stretch":
        y_aug = librosa.effects.time_stretch(y, rate=TIME_STRETCH_RATE)
        return fix_length(y_aug.astype(np.float32), TARGET_SAMPLES)

    if aug_name == "time_shift":
        shift = int(TIME_SHIFT_RATIO * len(y))
        if shift > 0:
            y_aug = np.concatenate([np.zeros(shift, dtype=y.dtype), y[:-shift]])
        else:
            y_aug = y.copy()
        return fix_length(y_aug.astype(np.float32), TARGET_SAMPLES)

    raise ValueError(f"Nieznana augmentacja: {aug_name}")


def audio_to_log_mel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # standaryzacja na poziomie próbki
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-8)

    # kanał dla CNN
    mel_db = mel_db[..., np.newaxis].astype(np.float32)
    return mel_db


# GENERATOR

class CNNDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        df: pd.DataFrame,
        label_encoder: LabelEncoder,
        batch_size: int = 32,
        training: bool = False,
        use_train_augmentation: bool = False,
        shuffle: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.batch_size = batch_size
        self.training = training
        self.use_train_augmentation = use_train_augmentation
        self.shuffle = shuffle

        self.base_len = len(self.df)

        # train: dokładnie 5 wariantów jak w klasycznych
        if self.training and self.use_train_augmentation:
            self.aug_list = TRAIN_AUGMENTATIONS
        else:
            self.aug_list = ["original"]

        self.total_len = self.base_len * len(self.aug_list)
        self.indices = np.arange(self.total_len)
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(self.total_len / self.batch_size)

    def __getitem__(self, index):
        batch_idx = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        X_batch = []
        y_batch = []

        for global_idx in batch_idx:
            sample_idx = global_idx % self.base_len
            aug_idx = global_idx // self.base_len
            aug_name = self.aug_list[aug_idx]

            row = self.df.iloc[sample_idx]
            y = load_audio(row["path"])
            y = augment_audio(y, aug_name)
            mel = audio_to_log_mel(y)

            X_batch.append(mel)
            y_batch.append(row["label"])

        X_batch = np.stack(X_batch, axis=0)
        y_encoded = self.label_encoder.transform(np.array(y_batch))
        y_batch = tf.keras.utils.to_categorical(y_encoded, num_classes=len(self.label_encoder.classes_))

        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# MODEL
def build_cnn_model(input_shape, n_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(CNN_FILTERS[0], CNN_KERNEL_SIZE, padding="same"),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(pool_size=CNN_POOL_SIZE),

        tf.keras.layers.Conv2D(CNN_FILTERS[1], CNN_KERNEL_SIZE, padding="same"),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPooling2D(pool_size=CNN_POOL_SIZE),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(CNN_DROPOUT),
        tf.keras.layers.Dense(CNN_DENSE_UNITS, activation="relu"),
        tf.keras.layers.Dropout(CNN_DROPOUT),
        tf.keras.layers.Dense(n_classes, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

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
    plt.title(f"Macierz pomyłek - CNN ({split_name})")
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


def plot_training_history(history):
    # loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoka")
    plt.ylabel("Loss")
    plt.title("Przebieg funkcji straty - CNN")
    plt.legend()
    plt.tight_layout()
    out1 = PLOTS_DIR / "cnn_loss_curve.png"
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"Zapisano wykres loss do: {out1}")

    # accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoka")
    plt.ylabel("Accuracy")
    plt.title("Przebieg accuracy - CNN")
    plt.legend()
    plt.tight_layout()
    out2 = PLOTS_DIR / "cnn_accuracy_curve.png"
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"Zapisano wykres accuracy do: {out2}")


def predict_generator(model, generator):
    pred_probs = model.predict(generator, verbose=1)
    y_pred = np.argmax(pred_probs, axis=1)
    y_true = []
    for i in range(len(generator)):
        _, y_batch = generator[i]
        y_true.extend(np.argmax(y_batch, axis=1).tolist())

    y_true = np.array(y_true[:len(y_pred)])
    return y_true, y_pred

def main():
    ensure_dirs()
    log("=== START PIPELINE CNN ===")

    #indeksy plików
    train_df = build_split_dataframe("train")
    val_df = build_split_dataframe("val")
    test_df = build_split_dataframe("test")

    save_split_dataframe(train_df, "train")
    save_split_dataframe(val_df, "val")
    save_split_dataframe(test_df, "test")

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["label"].values)

    class_names = list(label_encoder.classes_)
    n_classes = len(class_names)

    log(f"Liczba klas: {n_classes}")
    log(f"Train raw files: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    #generator train z 5 wariantami jak w klasycznych
    train_gen = CNNDataGenerator(
        train_df,
        label_encoder=label_encoder,
        batch_size=BATCH_SIZE,
        training=True,
        use_train_augmentation=USE_TRAIN_AUGMENTATION,
        shuffle=True,
    )

    val_gen = CNNDataGenerator(
        val_df,
        label_encoder=label_encoder,
        batch_size=BATCH_SIZE,
        training=False,
        use_train_augmentation=False,
        shuffle=False,
    )

    test_gen = CNNDataGenerator(
        test_df,
        label_encoder=label_encoder,
        batch_size=BATCH_SIZE,
        training=False,
        use_train_augmentation=False,
        shuffle=False,
    )

    #kształt wejścia
    sample_X, _ = train_gen[0]
    input_shape = sample_X.shape[1:]
    log(f"Kształt wejścia CNN: {input_shape}")

    #model
    model = build_cnn_model(input_shape=input_shape, n_classes=n_classes)
    model.summary(print_fn=lambda x: log(x))

    summary_txt = []
    model.summary(print_fn=lambda x: summary_txt.append(x))
    with open(REPORTS_DIR / "cnn_model_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_txt))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "best_cnn.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    #trening
    log("Rozpoczynam trening CNN...")
    train_start = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    train_time_s = time.time() - train_start
    log(f"Trening CNN zakończony. Czas treningu: {train_time_s:.2f} s")

    #zapis modelu końcowego
    final_model_path = MODELS_DIR / "cnn_final.keras"
    model.save(final_model_path)
    log(f"Zapisano model końcowy do: {final_model_path}")

    #wykresy uczenia
    plot_training_history(history)

    #ewaluacja
    summary_rows = []
    pred_times = {}

    for split_name, gen in [("val", val_gen), ("test", test_gen)]:
        log(f"Predykcja dla splitu: {split_name}")
        pred_start = time.time()
        y_true, y_pred = predict_generator(model, gen)
        pred_time_s = time.time() - pred_start
        pred_times[split_name] = pred_time_s
        log(f"Czas predykcji ({split_name}): {pred_time_s:.2f} s")

        metrics = compute_metrics(y_true, y_pred)
        metrics["split"] = split_name
        metrics["model"] = "CNN"
        metrics["n_samples"] = int(len(y_true))
        summary_rows.append(metrics)

        save_classification_report(y_true, y_pred, class_names, split_name)
        plot_confusion_matrix(y_true, y_pred, class_names, split_name)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = REPORTS_DIR / "cnn_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log(f"Zapisano podsumowanie metryk do: {summary_path}")

    # summary
    test_metrics = summary_df[summary_df["split"] == "test"].iloc[0].to_dict()
    val_metrics = summary_df[summary_df["split"] == "val"].iloc[0].to_dict()

    experiment_summary = pd.DataFrame([{
        "model": "CNN",
        "input_type": "mel_spectrogram",
        "n_classes": n_classes,
        "train_raw_audio_files": int(len(train_df)),
        "val_raw_audio_files": int(len(val_df)),
        "test_raw_audio_files": int(len(test_df)),
        "train_effective_samples": int(len(train_df) * len(TRAIN_AUGMENTATIONS)) if USE_TRAIN_AUGMENTATION else int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "input_shape": str(input_shape),
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
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "dropout": CNN_DROPOUT,
        "filters": "32,64",
        "kernel_size": "3x3",
        "pool_size": "2x2",
        "train_augmentation_enabled": USE_TRAIN_AUGMENTATION,
        "train_augmentations": ",".join(TRAIN_AUGMENTATIONS),
        "trainable_params": int(model.count_params()),
    }])

    experiment_summary_path = REPORTS_DIR / "cnn_experiment_summary.csv"
    experiment_summary.to_csv(experiment_summary_path, index=False)
    log(f"Zapisano rozszerzone podsumowanie eksperymentu do: {experiment_summary_path}")

    config = {
        "base_dir": str(BASE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "audio_params": {
            "sr": SR,
            "duration_s": DURATION,
            "target_samples": TARGET_SAMPLES,
        },
        "mel_params": {
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "fmin": FMIN,
            "fmax": FMAX,
        },
        "augmentation": {
            "enabled_train_only": USE_TRAIN_AUGMENTATION,
            "augmentations": TRAIN_AUGMENTATIONS,
            "noise_sigma": NOISE_SIGMA,
            "pitch_shift_steps": PITCH_SHIFT_STEPS,
            "time_stretch_rate": TIME_STRETCH_RATE,
            "time_shift_ratio": TIME_SHIFT_RATIO,
        },
        "cnn_params": {
            "filters": CNN_FILTERS,
            "kernel_size": CNN_KERNEL_SIZE,
            "pool_size": CNN_POOL_SIZE,
            "dropout": CNN_DROPOUT,
            "dense_units": CNN_DENSE_UNITS,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
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
    log("=== KONIEC PIPELINE CNN ===")


if __name__ == "__main__":
    main()