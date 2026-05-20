import json
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt



# KONFIGURACJA

INPUT_DIR = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\dzwieki_oczyszczone")
OUTPUT_DIR = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\class_count_report")

AUDIO_EXTENSIONS = {".wav"}
RARE_THRESHOLDS = [5, 10, 15, 20]
FIG_DPI = 150


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "class_count_log.txt"

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


# GŁÓWNA LOGIKA
def scan_class_counts(input_dir: Path) -> pd.DataFrame:
    rows = []

    class_dirs = [p for p in input_dir.iterdir() if p.is_dir()]
    class_dirs = sorted(class_dirs, key=lambda p: p.name.lower())

    for class_dir in class_dirs:
        class_name = class_dir.name.strip().lower()
        count = sum(1 for p in class_dir.glob("*.wav") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)

        rows.append({
            "class_name": class_name,
            "n_files": count
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("n_files", ascending=False).reset_index(drop=True)
    return df


def create_plots(df: pd.DataFrame, output_dir: Path):
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1.Wszystkie klasy
    plt.figure(figsize=(14, 10))
    plt.barh(df["class_name"], df["n_files"])
    plt.gca().invert_yaxis()
    plt.title("Liczba segmentów audio w poszczególnych klasach")
    plt.xlabel("Liczba plików .wav")
    plt.ylabel("Klasa")
    plt.tight_layout()
    plt.savefig(plots_dir / "all_class_counts.png", dpi=FIG_DPI)
    plt.close()

    # 2.Top 20 klas
    top20 = df.head(20).sort_values("n_files", ascending=True)
    plt.figure(figsize=(12, 8))
    plt.barh(top20["class_name"], top20["n_files"])
    plt.title("20 najliczniejszych klas")
    plt.xlabel("Liczba plików .wav")
    plt.ylabel("Klasa")
    plt.tight_layout()
    plt.savefig(plots_dir / "top20_class_counts.png", dpi=FIG_DPI)
    plt.close()

    # 3.20 najmniej licznych klas
    bottom20 = df.tail(20).sort_values("n_files", ascending=True)
    plt.figure(figsize=(12, 8))
    plt.barh(bottom20["class_name"], bottom20["n_files"])
    plt.title("20 najmniej licznych klas")
    plt.xlabel("Liczba plików .wav")
    plt.ylabel("Klasa")
    plt.tight_layout()
    plt.savefig(plots_dir / "bottom20_class_counts.png", dpi=FIG_DPI)
    plt.close()

    # 4.Histogram liczebności klas
    plt.figure(figsize=(10, 6))
    plt.hist(df["n_files"], bins=25)
    plt.title("Rozkład liczebności klas")
    plt.xlabel("Liczba plików w klasie")
    plt.ylabel("Liczba klas")
    plt.tight_layout()
    plt.savefig(plots_dir / "histogram_class_counts.png", dpi=FIG_DPI)
    plt.close()


def save_reports(df: pd.DataFrame, output_dir: Path):
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(reports_dir / "class_counts.csv", index=False, encoding="utf-8")

    summary = {
        "total_classes": int(len(df)),
        "total_files": int(df["n_files"].sum()),
        "mean_files_per_class": round(float(df["n_files"].mean()), 4),
        "median_files_per_class": round(float(df["n_files"].median()), 4),
        "min_files_per_class": int(df["n_files"].min()),
        "max_files_per_class": int(df["n_files"].max()),
    }

    for thr in RARE_THRESHOLDS:
        summary[f"classes_leq_{thr}"] = int((df["n_files"] <= thr).sum())
        summary[f"classes_lt_{thr}"] = int((df["n_files"] < thr).sum())

    with open(reports_dir / "class_count_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    with open(reports_dir / "class_count_summary.txt", "w", encoding="utf-8") as f:
        f.write("PODSUMOWANIE LICZEBNOŚCI KLAS\n")
        f.write("=" * 60 + "\n\n")

        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

        f.write("\nKLASY O NAJMNIEJSZEJ LICZEBNOŚCI (TOP 20 OD KOŃCA):\n")
        f.write("-" * 60 + "\n")
        for _, row in df.sort_values("n_files", ascending=True).head(20).iterrows():
            f.write(f"{row['class_name']}: {row['n_files']}\n")

        for thr in RARE_THRESHOLDS:
            subset = df[df["n_files"] < thr].sort_values("n_files", ascending=True)
            f.write(f"\nKLASY Z LICZBĄ PLIKÓW < {thr}:\n")
            f.write("-" * 60 + "\n")
            if subset.empty:
                f.write("Brak\n")
            else:
                for _, row in subset.iterrows():
                    f.write(f"{row['class_name']}: {row['n_files']}\n")


def main():
    setup_logging(OUTPUT_DIR)
    logging.info("=== START ANALIZY LICZEBNOŚCI KLAS ===")
    logging.info(f"Wejście: {INPUT_DIR}")
    logging.info(f"Wyjście: {OUTPUT_DIR}")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Brak katalogu wejściowego: {INPUT_DIR}")

    df = scan_class_counts(INPUT_DIR)

    if df.empty:
        raise RuntimeError("Nie znaleziono żadnych klas.")

    save_reports(df, OUTPUT_DIR)
    create_plots(df, OUTPUT_DIR)

    logging.info(f"Liczba klas: {len(df)}")
    logging.info(f"Liczba wszystkich plików: {df['n_files'].sum()}")
    logging.info(f"Średnia liczba plików na klasę: {df['n_files'].mean():.2f}")
    logging.info(f"Mediana liczby plików na klasę: {df['n_files'].median():.2f}")
    logging.info(f"Minimum: {df['n_files'].min()}")
    logging.info(f"Maksimum: {df['n_files'].max()}")

    for thr in RARE_THRESHOLDS:
        logging.info(f"Klasy z liczbą plików < {thr}: {(df['n_files'] < thr).sum()}")

    logging.info("=== ZAKOŃCZONO ANALIZĘ LICZEBNOŚCI KLAS ===")
    logging.info(f"- {OUTPUT_DIR / 'reports' / 'class_counts.csv'}")
    logging.info(f"- {OUTPUT_DIR / 'reports' / 'class_count_summary.txt'}")
    logging.info(f"- {OUTPUT_DIR / 'reports' / 'class_count_summary.json'}")
    logging.info(f"- {OUTPUT_DIR / 'plots'}")


if __name__ == "__main__":
    main()