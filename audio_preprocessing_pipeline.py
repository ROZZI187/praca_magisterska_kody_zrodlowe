import os
import json
import math
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm



# KONFIGURACJA

INPUT_DIR = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\soundscape_data")
OUTPUT_DIR = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\dzwieki_oczyszczone")

ANNOTATIONS_FILE = INPUT_DIR / "annotations_clean.csv"
TARGET_SR = 22050
TARGET_DURATION_SEC = 5.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION_SEC)

AUDIO_EXT = ".wav"
NORMALIZE_AUDIO = True
PEAK_NORM_LEVEL = 0.98

# Jeśli True: pliki będą rozdzielane do podfolderów wg gatunku

SAVE_IN_SPECIES_SUBFOLDERS = True

# wykresy
TOP_N_SPECIES_PLOT = 20
FIG_DPI = 150


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "preprocessing_log.txt"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


# NARZĘDZIA
def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def sanitize_label(text: str) -> str:
    """Uproszczenie nazwy gatunku do bezpiecznej nazwy pliku."""
    text = str(text).strip().lower()
    allowed = []
    for ch in text:
        if ch.isalnum() or ch in ("_", "-"):
            allowed.append(ch)
        else:
            allowed.append("_")
    cleaned = "".join(allowed)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unknown"


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """
    Wejście:
      - (n,) lub
      - (n, channels)
    Wyjście:
      - (n,) mono
    """
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return np.mean(audio, axis=1).astype(np.float32)


def normalize_peak(audio: np.ndarray, peak_level: float = 0.98) -> np.ndarray:
    max_val = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if max_val > 0:
        audio = audio / max_val * peak_level
    return audio.astype(np.float32)


def pad_or_trim(audio: np.ndarray, target_samples: int) -> tuple[np.ndarray, str]:
    """
    Zwraca:
      audio_fixed, operation
    operation in {"padded", "trimmed", "unchanged"}
    """
    n = len(audio)
    if n < target_samples:
        padded = np.zeros(target_samples, dtype=np.float32)
        padded[:n] = audio
        return padded, "padded"
    elif n > target_samples:
        return audio[:target_samples].astype(np.float32), "trimmed"
    else:
        return audio.astype(np.float32), "unchanged"


def create_plots(df_meta: pd.DataFrame, output_dir: Path):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Histogram długości oryginalnych segmentów
    plt.figure(figsize=(10, 6))
    plt.hist(df_meta["original_duration_sec"], bins=40)
    plt.title("Rozkład długości oryginalnych segmentów")
    plt.xlabel("Długość segmentu [s]")
    plt.ylabel("Liczba segmentów")
    plt.tight_layout()
    plt.savefig(plots_dir / "histogram_dlugosci_segmentow.png", dpi=FIG_DPI)
    plt.close()

    # 2. Top gatunki
    species_counts = df_meta["species_code"].value_counts().head(TOP_N_SPECIES_PLOT)
    plt.figure(figsize=(12, 7))
    species_counts.sort_values().plot(kind="barh")
    plt.title(f"Top {TOP_N_SPECIES_PLOT} gatunków wg liczby segmentów")
    plt.xlabel("Liczba segmentów")
    plt.ylabel("Gatunek")
    plt.tight_layout()
    plt.savefig(plots_dir / "top_gatunki_segmenty.png", dpi=FIG_DPI)
    plt.close()

    # 3. Liczba segmentów na plik źródłowy
    file_counts = df_meta["source_filename"].value_counts()
    plt.figure(figsize=(10, 6))
    plt.hist(file_counts.values, bins=30)
    plt.title("Rozkład liczby segmentów przypadających na nagranie źródłowe")
    plt.xlabel("Liczba segmentów w nagraniu")
    plt.ylabel("Liczba nagrań")
    plt.tight_layout()
    plt.savefig(plots_dir / "histogram_segmentow_na_nagranie.png", dpi=FIG_DPI)
    plt.close()

    # 4. Typ operacji: padding / trimming / unchanged
    op_counts = df_meta["length_adjustment"].value_counts()
    plt.figure(figsize=(8, 5))
    op_counts.plot(kind="bar")
    plt.title("Operacje dopasowania długości segmentów")
    plt.xlabel("Typ operacji")
    plt.ylabel("Liczba segmentów")
    plt.tight_layout()
    plt.savefig(plots_dir / "operacje_dopasowania_dlugosci.png", dpi=FIG_DPI)
    plt.close()


def save_reports(df_meta: pd.DataFrame, output_dir: Path):
    meta_csv = output_dir / "segment_metadata.csv"
    df_meta.to_csv(meta_csv, index=False, encoding="utf-8")

    # podsumowanie per gatunek
    species_summary = (
        df_meta.groupby("species_code")
        .agg(
            segments=("species_code", "size"),
            mean_original_duration_sec=("original_duration_sec", "mean"),
            min_original_duration_sec=("original_duration_sec", "min"),
            max_original_duration_sec=("original_duration_sec", "max"),
        )
        .sort_values("segments", ascending=False)
        .reset_index()
    )
    species_summary.to_csv(output_dir / "species_summary.csv", index=False, encoding="utf-8")

    # podsumowanie ogólne
    summary = {
        "input_dir": str(INPUT_DIR),
        "output_dir": str(output_dir),
        "annotations_file": str(ANNOTATIONS_FILE),
        "target_sample_rate_hz": TARGET_SR,
        "target_duration_sec": TARGET_DURATION_SEC,
        "total_segments_saved": int(len(df_meta)),
        "total_source_recordings_used": int(df_meta["source_filename"].nunique()),
        "total_species": int(df_meta["species_code"].nunique()),
        "mean_original_duration_sec": round(float(df_meta["original_duration_sec"].mean()), 4),
        "median_original_duration_sec": round(float(df_meta["original_duration_sec"].median()), 4),
        "min_original_duration_sec": round(float(df_meta["original_duration_sec"].min()), 4),
        "max_original_duration_sec": round(float(df_meta["original_duration_sec"].max()), 4),
        "padded_segments": int((df_meta["length_adjustment"] == "padded").sum()),
        "trimmed_segments": int((df_meta["length_adjustment"] == "trimmed").sum()),
        "unchanged_segments": int((df_meta["length_adjustment"] == "unchanged").sum()),
    }

    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    with open(output_dir / "dataset_summary.txt", "w", encoding="utf-8") as f:
        f.write("PODSUMOWANIE PRZETWARZANIA DANYCH AUDIO\n")
        f.write("=" * 50 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

        f.write("\nTOP 20 GATUNKÓW WG LICZBY SEGMENTÓW\n")
        f.write("-" * 50 + "\n")
        top_species = df_meta["species_code"].value_counts().head(20)
        for species, count in top_species.items():
            f.write(f"{species}: {count}\n")


def load_annotations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    expected_cols = [
        "Filename",
        "Start Time (s)",
        "End Time (s)",
        "Low Freq (Hz)",
        "High Freq (Hz)",
        "Species eBird Code"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Brakuje kolumn w annotations_clean.csv: {missing}")

    df = df.rename(columns={
        "Filename": "filename",
        "Start Time (s)": "start_time_sec",
        "End Time (s)": "end_time_sec",
        "Low Freq (Hz)": "low_freq_hz",
        "High Freq (Hz)": "high_freq_hz",
        "Species eBird Code": "species_code",
    })

    df["filename"] = df["filename"].astype(str).str.strip()
    df["species_code"] = df["species_code"].astype(str).str.strip().str.lower()

    numeric_cols = ["start_time_sec", "end_time_sec", "low_freq_hz", "high_freq_hz"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # odrzucenie błędnych rekordów
    before = len(df)
    df = df.dropna(subset=["filename", "start_time_sec", "end_time_sec", "species_code"]).copy()
    df = df[df["end_time_sec"] > df["start_time_sec"]].copy()
    dropped = before - len(df)

    logging.info(f"Wczytano annotations_clean.csv: {before} rekordów.")
    logging.info(f"Odrzucono błędne rekordy: {dropped}.")
    logging.info(f"Pozostało poprawnych rekordów: {len(df)}.")

    # numer segmentu w obrębie pliku źródłowego
    df = df.sort_values(["filename", "start_time_sec", "end_time_sec"]).reset_index(drop=True)
    df["segment_idx_in_file"] = df.groupby("filename").cumcount() + 1

    return df


def process_group(source_file: Path, group_df: pd.DataFrame, output_dir: Path) -> list[dict]:
    """
    laduje jedno nagranie źródłowe i przetwarza wszystkie segmenty z tego nagrania.
    """
    metadata_rows = []

    if not source_file.exists():
        logging.warning(f"Brak pliku audio: {source_file.name}")
        return metadata_rows

    try:
        audio, sr = sf.read(source_file, always_2d=True)
        audio = ensure_mono(audio)
    except Exception as e:
        logging.exception(f"Nie udało się wczytać pliku {source_file.name}: {e}")
        return metadata_rows

    for _, row in group_df.iterrows():
        start_sec = float(row["start_time_sec"])
        end_sec = float(row["end_time_sec"])
        low_freq = safe_float(row["low_freq_hz"], 0.0)
        high_freq = safe_float(row["high_freq_hz"], 0.0)
        species_code = sanitize_label(row["species_code"])
        seg_idx = int(row["segment_idx_in_file"])

        start_sample = max(0, int(round(start_sec * sr)))
        end_sample = min(len(audio), int(round(end_sec * sr)))

        if end_sample <= start_sample:
            logging.warning(
                f"Pominięto pusty segment | plik={source_file.name} | "
                f"start={start_sec:.3f} | end={end_sec:.3f}"
            )
            continue

        segment = audio[start_sample:end_sample]
        original_duration = len(segment) / sr

        # resampling do wspólnego sr
        if sr != TARGET_SR:
            segment = librosa.resample(segment, orig_sr=sr, target_sr=TARGET_SR)

        # normalizacja amplitudy
        if NORMALIZE_AUDIO:
            segment = normalize_peak(segment, PEAK_NORM_LEVEL)

        # dopasowanie do stałej długości
        segment_fixed, adjustment = pad_or_trim(segment, TARGET_SAMPLES)

        base_name = source_file.stem.split("_")[0] + "_" + source_file.stem.split("_")[1]
        parts = source_file.stem.split("_")
        if len(parts) >= 2:
            recording_id = f"{parts[0]}_{parts[1]}"
        else:
            recording_id = source_file.stem

        out_filename = f"{recording_id}_seg{seg_idx:04d}_{species_code}{AUDIO_EXT}"

        if SAVE_IN_SPECIES_SUBFOLDERS:
            species_dir = output_dir / species_code
            species_dir.mkdir(parents=True, exist_ok=True)
            out_path = species_dir / out_filename
        else:
            out_path = output_dir / out_filename

        try:
            sf.write(out_path, segment_fixed, TARGET_SR, subtype="PCM_16")
        except Exception as e:
            logging.exception(f"Nie udało się zapisać pliku {out_path.name}: {e}")
            continue

        metadata_rows.append({
            "source_filename": source_file.name,
            "recording_id": recording_id,
            "segment_idx_in_file": seg_idx,
            "output_filename": out_filename,
            "output_path": str(out_path),
            "species_code": species_code,
            "start_time_sec": round(start_sec, 4),
            "end_time_sec": round(end_sec, 4),
            "original_duration_sec": round(original_duration, 4),
            "target_duration_sec": TARGET_DURATION_SEC,
            "source_sample_rate_hz": sr,
            "target_sample_rate_hz": TARGET_SR,
            "length_adjustment": adjustment,
            "low_freq_hz": low_freq,
            "high_freq_hz": high_freq,
        })

    return metadata_rows


def main():
    setup_logging(OUTPUT_DIR)
    logging.info("=== START PRZETWARZANIA NAGRAŃ ===")
    logging.info(f"Wejście: {INPUT_DIR}")
    logging.info(f"Wyjście: {OUTPUT_DIR}")
    logging.info(f"Plik adnotacji: {ANNOTATIONS_FILE}")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Nie istnieje katalog wejściowy: {INPUT_DIR}")
    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(f"Nie istnieje plik: {ANNOTATIONS_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_ann = load_annotations(ANNOTATIONS_FILE)

    # grupowanie po pliku źródłowym, aby każdy FLAC wczytywać raz
    grouped = df_ann.groupby("filename", sort=True)

    logging.info(f"Liczba unikalnych nagrań źródłowych użytych w adnotacjach: {len(grouped)}")
    logging.info(f"Liczba wszystkich segmentów do przygotowania: {len(df_ann)}")

    all_meta = []
    missing_files = []
    processed_files = 0

    for filename, group_df in tqdm(grouped, desc="Przetwarzanie plików FLAC"):
        source_file = INPUT_DIR / filename

        if not source_file.exists():
            missing_files.append(filename)
            logging.warning(f"Nie znaleziono pliku: {filename}")
            continue

        rows = process_group(source_file, group_df, OUTPUT_DIR)
        all_meta.extend(rows)
        processed_files += 1

        if processed_files % 20 == 0:
            logging.info(
                f"Postęp: przetworzono {processed_files} plików źródłowych, "
                f"zapisano dotąd {len(all_meta)} segmentów."
            )

    if not all_meta:
        logging.error("Nie zapisano żadnego segmentu. Sprawdź dane wejściowe.")
        return

    df_meta = pd.DataFrame(all_meta)
    df_meta = df_meta.sort_values(["source_filename", "segment_idx_in_file"]).reset_index(drop=True)

    save_reports(df_meta, OUTPUT_DIR)
    create_plots(df_meta, OUTPUT_DIR)

    if missing_files:
        with open(OUTPUT_DIR / "missing_files.txt", "w", encoding="utf-8") as f:
            for item in missing_files:
                f.write(item + "\n")

    logging.info("=== ZAKOŃCZONO PRZETWARZANIE ===")
    logging.info(f"Zapisane segmenty: {len(df_meta)}")
    logging.info(f"Liczba gatunków: {df_meta['species_code'].nunique()}")
    logging.info(f"Liczba wykorzystanych nagrań źródłowych: {df_meta['source_filename'].nunique()}")
    logging.info(f"Średnia długość oryginalnych segmentów: {df_meta['original_duration_sec'].mean():.4f} s")
    logging.info(f"Mediana długości oryginalnych segmentów: {df_meta['original_duration_sec'].median():.4f} s")
    logging.info(f"Padding: {(df_meta['length_adjustment'] == 'padded').sum()}")
    logging.info(f"Trim: {(df_meta['length_adjustment'] == 'trimmed').sum()}")
    logging.info(f"Bez zmian długości: {(df_meta['length_adjustment'] == 'unchanged').sum()}")

    logging.info("Wygenerowane pliki końcowe:")
    logging.info(f"- {OUTPUT_DIR / 'segment_metadata.csv'}")
    logging.info(f"- {OUTPUT_DIR / 'species_summary.csv'}")
    logging.info(f"- {OUTPUT_DIR / 'dataset_summary.txt'}")
    logging.info(f"- {OUTPUT_DIR / 'dataset_summary.json'}")
    logging.info(f"- {OUTPUT_DIR / 'plots'}")


if __name__ == "__main__":
    main()