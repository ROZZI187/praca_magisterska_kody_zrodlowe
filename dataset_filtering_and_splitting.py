import json
import math
import shutil
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm



# KONFIGURACJA
INPUT_DIR = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\dzwieki_oczyszczone")
OUTPUT_DIR = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\data_splits")

AUDIO_EXTENSIONS = {".wav"}
MIN_CLASS_SIZE = 20

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CREATE_SPLIT_FOLDERS = True
USE_HARDLINKS_IF_POSSIBLE = True

RANDOM_STATE = 42
FIG_DPI = 150


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "split_log.txt"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# WCZYTANIE DANYCH
def scan_dataset(input_dir: Path) -> pd.DataFrame:
    rows = []

    class_dirs = [p for p in input_dir.iterdir() if p.is_dir()]
    class_dirs = sorted(class_dirs, key=lambda p: p.name.lower())

    for class_dir in tqdm(class_dirs, desc="Skanowanie klas"):
        class_name = class_dir.name.strip().lower()

        for file_path in class_dir.glob("*.wav"):
            if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS:
                rows.append({
                    "filepath": str(file_path),
                    "filename": file_path.name,
                    "class_name": class_name,
                })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("Nie znaleziono żadnych plików .wav w katalogu wejściowym.")

    return df


def build_class_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby("class_name")
        .size()
        .reset_index(name="n_files")
        .sort_values("n_files", ascending=False)
        .reset_index(drop=True)
    )
    return counts

def filter_classes_by_threshold(df: pd.DataFrame, min_class_size: int):
    class_counts = build_class_counts(df)

    kept_classes = class_counts[class_counts["n_files"] >= min_class_size]["class_name"].tolist()
    removed_df = class_counts[class_counts["n_files"] < min_class_size].copy()

    filtered_df = df[df["class_name"].isin(kept_classes)].copy()

    return filtered_df, class_counts, removed_df


def split_single_class(df_class: pd.DataFrame) -> pd.DataFrame:
    df_class = df_class.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    n = len(df_class)

    n_train = int(math.floor(n * TRAIN_RATIO))
    n_val = int(math.floor(n * VAL_RATIO))
    n_test = n - n_train - n_val

    if n_val < 1:
        n_val = 1
    if n_test < 1:
        n_test = 1
    n_train = n - n_val - n_test

    if n_train < 1:
        raise ValueError(f"Klasa ma zbyt mało danych po korekcie splitu: n={n}")

    splits = (
        ["train"] * n_train +
        ["val"] * n_val +
        ["test"] * n_test
    )

    if len(splits) != n:
        raise ValueError("Błąd: długość splitu nie zgadza się z liczbą plików w klasie.")

    df_class = df_class.copy()
    df_class["split"] = splits
    return df_class


def stratified_split_per_class(df: pd.DataFrame) -> pd.DataFrame:
    parts = []

    for class_name, df_class in df.groupby("class_name", sort=True):
        parts.append(split_single_class(df_class.reset_index(drop=True)))

    result = pd.concat(parts, ignore_index=True)
    return result


# EKSPORT PLIKÓW
def safe_link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return "exists"

    if USE_HARDLINKS_IF_POSSIBLE:
        try:
            import os
            os.link(src, dst)
            return "hardlink"
        except Exception:
            pass

    shutil.copy2(src, dst)
    return "copy"


def create_split_folders(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    export_modes = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tworzenie folderów splitów"):
        src = Path(row["filepath"])
        dst = output_dir / row["split"] / row["class_name"] / row["filename"]
        mode = safe_link_or_copy(src, dst)
        export_modes.append(mode)

    df = df.copy()
    df["file_export_mode"] = export_modes
    return df


# RAPORTY

def save_manifests(df: pd.DataFrame, output_dir: Path):
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(manifests_dir / "all_files_with_splits.csv", index=False, encoding="utf-8")

    for split_name in ["train", "val", "test"]:
        df[df["split"] == split_name].to_csv(
            manifests_dir / f"{split_name}_manifest.csv",
            index=False,
            encoding="utf-8"
        )


def create_reports(df_all: pd.DataFrame, df_filtered: pd.DataFrame, removed_df: pd.DataFrame, output_dir: Path):
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # pełne liczebności klas przed filtrowaniem
    full_counts = build_class_counts(df_all)
    full_counts.to_csv(reports_dir / "class_counts_before_filtering.csv", index=False, encoding="utf-8")

    # klasy usunięte
    removed_df.to_csv(reports_dir / "removed_classes_below_threshold.csv", index=False, encoding="utf-8")

    # podsumowanie splitów
    split_summary = (
        df_filtered.groupby("split")
        .agg(
            n_files=("filename", "count"),
            n_classes=("class_name", "nunique"),
        )
        .reset_index()
    )

    total_files_filtered = len(df_filtered)
    split_summary["files_pct"] = (split_summary["n_files"] / total_files_filtered * 100).round(2)
    split_summary.to_csv(reports_dir / "split_summary.csv", index=False, encoding="utf-8")

    # rozkład klas w splitach
    class_distribution = (
        df_filtered.groupby(["class_name", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for split_name in ["train", "val", "test"]:
        if split_name not in class_distribution.columns:
            class_distribution[split_name] = 0

    class_distribution["total"] = (
        class_distribution["train"] +
        class_distribution["val"] +
        class_distribution["test"]
    )
    class_distribution = class_distribution.sort_values("total", ascending=False)
    class_distribution.to_csv(reports_dir / "class_distribution_by_split.csv", index=False, encoding="utf-8")

    summary = {
        "min_class_size": MIN_CLASS_SIZE,
        "total_classes_before_filtering": int(df_all["class_name"].nunique()),
        "total_classes_after_filtering": int(df_filtered["class_name"].nunique()),
        "removed_classes_count": int(len(removed_df)),
        "total_files_before_filtering": int(len(df_all)),
        "total_files_after_filtering": int(len(df_filtered)),
        "removed_files_count": int(len(df_all) - len(df_filtered)),
        "removed_files_pct": round((len(df_all) - len(df_filtered)) / len(df_all) * 100, 4),
        "split_summary": split_summary.to_dict(orient="records"),
    }

    with open(reports_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    with open(reports_dir / "split_summary.txt", "w", encoding="utf-8") as f:
        f.write("PODSUMOWANIE FILTRACJI I PODZIAŁU DANYCH\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Minimalna liczba segmentów w klasie: {MIN_CLASS_SIZE}\n")
        f.write(f"Liczba klas przed filtrowaniem: {df_all['class_name'].nunique()}\n")
        f.write(f"Liczba klas po filtrowaniu: {df_filtered['class_name'].nunique()}\n")
        f.write(f"Liczba klas usuniętych: {len(removed_df)}\n")
        f.write(f"Liczba plików przed filtrowaniem: {len(df_all)}\n")
        f.write(f"Liczba plików po filtrowaniu: {len(df_filtered)}\n")
        f.write(f"Liczba usuniętych plików: {len(df_all) - len(df_filtered)}\n")
        f.write(f"Odsetek usuniętych plików: {summary['removed_files_pct']}%\n\n")

        f.write("PODZIAŁ 70/15/15:\n")
        for _, row in split_summary.iterrows():
            f.write(
                f"- {row['split']}: pliki={row['n_files']} ({row['files_pct']}%), "
                f"klasy={row['n_classes']}\n"
            )

        f.write("\nUSUNIĘTE KLASY (PONIŻEJ PROGU):\n")
        f.write("-" * 70 + "\n")
        if removed_df.empty:
            f.write("Brak\n")
        else:
            for _, row in removed_df.sort_values("n_files", ascending=True).iterrows():
                f.write(f"{row['class_name']}: {row['n_files']}\n")

        f.write("\nTOP 20 KLAS PO FILTRACJI:\n")
        f.write("-" * 70 + "\n")
        top20 = class_distribution.head(20)
        for _, row in top20.iterrows():
            f.write(
                f"{row['class_name']}: total={row['total']}, "
                f"train={row['train']}, val={row['val']}, test={row['test']}\n"
            )


def create_plots(df_all: pd.DataFrame, df_filtered: pd.DataFrame, removed_df: pd.DataFrame, output_dir: Path):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. liczebność klas przed filtrowaniem
    counts_all = build_class_counts(df_all).sort_values("n_files", ascending=True)

    plt.figure(figsize=(12, 16))
    plt.barh(counts_all["class_name"], counts_all["n_files"])
    plt.title("Liczba segmentów audio w klasach przed filtrowaniem")
    plt.xlabel("Liczba plików")
    plt.ylabel("Klasa")
    plt.tight_layout()
    plt.savefig(plots_dir / "class_counts_before_filtering.png", dpi=FIG_DPI)
    plt.close()

    # 2. liczebność klas po filtrowaniu
    counts_filtered = build_class_counts(df_filtered).sort_values("n_files", ascending=True)

    plt.figure(figsize=(12, 14))
    plt.barh(counts_filtered["class_name"], counts_filtered["n_files"])
    plt.title("Liczba segmentów audio w klasach po filtrowaniu")
    plt.xlabel("Liczba plików")
    plt.ylabel("Klasa")
    plt.tight_layout()
    plt.savefig(plots_dir / "class_counts_after_filtering.png", dpi=FIG_DPI)
    plt.close()

    # 3. pliki per split
    split_counts = df_filtered["split"].value_counts().reindex(["train", "val", "test"])
    plt.figure(figsize=(8, 5))
    split_counts.plot(kind="bar")
    plt.title("Liczba segmentów audio w poszczególnych splitach")
    plt.xlabel("Split")
    plt.ylabel("Liczba plików")
    plt.tight_layout()
    plt.savefig(plots_dir / "split_counts_files.png", dpi=FIG_DPI)
    plt.close()

    # 4. top 20 klas po filtracji z rozkładem splitów
    class_distribution = (
        df_filtered.groupby(["class_name", "split"])
        .size()
        .unstack(fill_value=0)
    )
    for split_name in ["train", "val", "test"]:
        if split_name not in class_distribution.columns:
            class_distribution[split_name] = 0

    class_distribution["total"] = class_distribution.sum(axis=1)
    top20 = class_distribution.sort_values("total", ascending=False).head(20).sort_values("total")

    plt.figure(figsize=(12, 8))
    plt.barh(top20.index, top20["train"], label="train")
    plt.barh(top20.index, top20["val"], left=top20["train"], label="val")
    plt.barh(top20.index, top20["test"], left=top20["train"] + top20["val"], label="test")
    plt.title("Rozkład splitów dla 20 najliczniejszych klas")
    plt.xlabel("Liczba plików")
    plt.ylabel("Klasa")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "top20_class_distribution_by_split.png", dpi=FIG_DPI)
    plt.close()

    # 5. klasy usunięte
    if not removed_df.empty:
        removed_sorted = removed_df.sort_values("n_files", ascending=True)

        plt.figure(figsize=(10, 8))
        plt.barh(removed_sorted["class_name"], removed_sorted["n_files"])
        plt.title(f"Klasy usunięte z powodu liczebności < {MIN_CLASS_SIZE}")
        plt.xlabel("Liczba plików")
        plt.ylabel("Klasa")
        plt.tight_layout()
        plt.savefig(plots_dir / "removed_classes_below_threshold.png", dpi=FIG_DPI)
        plt.close()

def main():
    setup_logging(OUTPUT_DIR)
    logging.info("=== START FILTRACJI I PODZIAŁU DANYCH ===")
    logging.info(f"Wejście: {INPUT_DIR}")
    logging.info(f"Wyjście: {OUTPUT_DIR}")
    logging.info(f"Minimalna liczba plików w klasie: {MIN_CLASS_SIZE}")
    logging.info(f"Podział: train={TRAIN_RATIO:.0%}, val={VAL_RATIO:.0%}, test={TEST_RATIO:.0%}")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Brak katalogu wejściowego: {INPUT_DIR}")

    df_all = scan_dataset(INPUT_DIR)
    logging.info(f"Wczytano wszystkich plików: {len(df_all)}")
    logging.info(f"Liczba klas przed filtrowaniem: {df_all['class_name'].nunique()}")

    df_filtered_base, class_counts, removed_df = filter_classes_by_threshold(df_all, MIN_CLASS_SIZE)

    logging.info(f"Liczba klas po filtrowaniu: {df_filtered_base['class_name'].nunique()}")
    logging.info(f"Liczba usuniętych klas: {len(removed_df)}")
    logging.info(f"Liczba plików po filtrowaniu: {len(df_filtered_base)}")
    logging.info(f"Liczba usuniętych plików: {len(df_all) - len(df_filtered_base)}")

    df_filtered = stratified_split_per_class(df_filtered_base)

    if CREATE_SPLIT_FOLDERS:
        df_filtered = create_split_folders(df_filtered, OUTPUT_DIR)

    save_manifests(df_filtered, OUTPUT_DIR)
    create_reports(df_all, df_filtered, removed_df, OUTPUT_DIR)
    create_plots(df_all, df_filtered, removed_df, OUTPUT_DIR)

    logging.info("=== ZAKOŃCZONO FILTRACJĘ I PODZIAŁ DANYCH ===")
    for split_name in ["train", "val", "test"]:
        part = df_filtered[df_filtered["split"] == split_name]
        logging.info(
            f"{split_name.upper()}: pliki={len(part)}, klasy={part['class_name'].nunique()}"
        )

    logging.info("Wygenerowane zasoby:")
    logging.info(f"- {OUTPUT_DIR / 'manifests'}")
    logging.info(f"- {OUTPUT_DIR / 'reports'}")
    logging.info(f"- {OUTPUT_DIR / 'plots'}")
    if CREATE_SPLIT_FOLDERS:
        logging.info(f"- {OUTPUT_DIR / 'train'}")
        logging.info(f"- {OUTPUT_DIR / 'val'}")
        logging.info(f"- {OUTPUT_DIR / 'test'}")


if __name__ == "__main__":
    main()