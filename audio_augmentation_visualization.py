import logging
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib import gridspec


TRAIN_DIR = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\data_splits\train")
OUTPUT_DIR = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\augmentation_figures")

SAMPLE_WAV = None

TARGET_SR = 22050
TARGET_DURATION_SEC = 5.0
TARGET_LEN = int(TARGET_SR * TARGET_DURATION_SEC)

# parametry augmentacji
NOISE_STD_RATIO = 0.005
PITCH_SHIFT_STEPS = 1.0
TIME_STRETCH_RATE = 0.90
TIME_SHIFT_RATIO = 0.10

SAVE_AUDIO_EXAMPLES = True
FIG_DPI = 180

# parametry spektrogramu
N_FFT = 2048
HOP_LENGTH = 512
TOP_DB = 80

def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def find_sample_wav(train_dir: Path) -> Path:
    wavs = sorted(train_dir.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"Nie znaleziono plików WAV w katalogu: {train_dir}")
    return wavs[0]


def fix_length(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) < target_len:
        out = np.zeros(target_len, dtype=np.float32)
        out[:len(y)] = y
        return out
    return y[:target_len].astype(np.float32)


def normalize_peak(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_amp = np.max(np.abs(y)) if len(y) else 0.0
    if max_amp > 0:
        y = y / max_amp * peak
    return y.astype(np.float32)


def load_audio_mono_fixed(path: Path, sr: int, target_len: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = fix_length(y, target_len)
    y = normalize_peak(y)
    return y


# AUGMENTACJE
def add_noise(y: np.ndarray, std_ratio: float = 0.005) -> np.ndarray:
    amp = np.max(np.abs(y)) if len(y) else 1.0
    noise = np.random.normal(0, amp * std_ratio, size=len(y))
    y_out = y + noise
    return normalize_peak(y_out.astype(np.float32))


def pitch_shift_audio(y: np.ndarray, sr: int, steps: float) -> np.ndarray:
    y_out = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
    y_out = fix_length(y_out, len(y))
    return normalize_peak(y_out)


def time_stretch_audio(y: np.ndarray, rate: float) -> np.ndarray:
    y_out = librosa.effects.time_stretch(y, rate=rate)
    y_out = fix_length(y_out, len(y))
    return normalize_peak(y_out)


def time_shift_audio(y: np.ndarray, shift_ratio: float) -> np.ndarray:
    shift = int(len(y) * shift_ratio)
    y_out = np.roll(y, shift)
    return normalize_peak(y_out.astype(np.float32))


# SPEKTROGRAM
def compute_mel_db(y: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max, top_db=TOP_DB)
    return S_db


# WAVEFORM
def plot_waveform(ax, y: np.ndarray, sr: int, title: str):
    t = np.arange(len(y)) / sr
    ax.plot(t, y, linewidth=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, len(y) / sr)
    ax.set_ylabel("Amplituda")
    ax.grid(True, alpha=0.25)


def generate_waveform_figure(
    original: np.ndarray,
    noise: np.ndarray,
    pitch: np.ndarray,
    stretch: np.ndarray,
    shift: np.ndarray,
    sr: int,
    output_path: Path
):
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)

    plot_waveform(axes[0], original, sr, "Oryginalny segment")
    plot_waveform(axes[1], noise, sr, "Segment po dodaniu szumu")
    plot_waveform(axes[2], pitch, sr, "Segment po zmianie wysokości tonu (pitch shift)")
    plot_waveform(axes[3], stretch, sr, "Segment po zmianie tempa (time stretch)")
    plot_waveform(axes[4], shift, sr, "Segment po przesunięciu w czasie (time shift)")

    axes[-1].set_xlabel("Czas [s]")
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def generate_spectrogram_figure(
    original: np.ndarray,
    noise: np.ndarray,
    pitch: np.ndarray,
    stretch: np.ndarray,
    shift: np.ndarray,
    sr: int,
    output_path: Path
):
    signals = [
        ("Oryginalny segment", original),
        ("Segment po dodaniu szumu", noise),
        ("Segment po zmianie wysokości tonu (pitch shift)", pitch),
        ("Segment po zmianie tempa (time stretch)", stretch),
        ("Segment po przesunięciu w czasie (time shift)", shift),
    ]

    spectrograms = [compute_mel_db(sig, sr) for _, sig in signals]

    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(
        nrows=5,
        ncols=2,
        width_ratios=[40, 1.8],
        hspace=0.42,
        wspace=0.08
    )

    axes = [fig.add_subplot(gs[i, 0]) for i in range(5)]
    cax = fig.add_subplot(gs[:, 1])

    img = None
    for ax, (title, _), S_db in zip(axes, signals, spectrograms):
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            ax=ax
        )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Hz")
        ax.label_outer()

    axes[-1].set_xlabel("Czas [s]")

    cbar = fig.colorbar(img, cax=cax)
    cbar.set_label("Poziom energii [dB]")

    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

def main():
    setup_logging(OUTPUT_DIR)
    logging.info("=== START GENEROWANIA RYSUNKÓW AUGMENTACJI V2 ===")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if SAMPLE_WAV is not None:
        sample_path = Path(SAMPLE_WAV)
    else:
        sample_path = find_sample_wav(TRAIN_DIR)

    logging.info(f"Wybrany plik przykładowy: {sample_path}")

    original = load_audio_mono_fixed(sample_path, TARGET_SR, TARGET_LEN)

    np.random.seed(42)
    aug_noise = add_noise(original, NOISE_STD_RATIO)
    aug_pitch = pitch_shift_audio(original, TARGET_SR, PITCH_SHIFT_STEPS)
    aug_stretch = time_stretch_audio(original, TIME_STRETCH_RATE)
    aug_shift = time_shift_audio(original, TIME_SHIFT_RATIO)

    if SAVE_AUDIO_EXAMPLES:
        sf.write(OUTPUT_DIR / "example_original.wav", original, TARGET_SR)
        sf.write(OUTPUT_DIR / "example_noise.wav", aug_noise, TARGET_SR)
        sf.write(OUTPUT_DIR / "example_pitch_shift.wav", aug_pitch, TARGET_SR)
        sf.write(OUTPUT_DIR / "example_time_stretch.wav", aug_stretch, TARGET_SR)
        sf.write(OUTPUT_DIR / "example_time_shift.wav", aug_shift, TARGET_SR)

    generate_waveform_figure(
        original, aug_noise, aug_pitch, aug_stretch, aug_shift,
        TARGET_SR,
        OUTPUT_DIR / "rys_4_10a_waveformy_augmentacji.png"
    )

    generate_spectrogram_figure(
        original, aug_noise, aug_pitch, aug_stretch, aug_shift,
        TARGET_SR,
        OUTPUT_DIR / "rys_4_10b_spektrogramy_augmentacji.png"
    )

    logging.info("Wygenerowano pliki:")
    logging.info(f"- {OUTPUT_DIR / 'rys_4_10a_waveformy_augmentacji.png'}")
    logging.info(f"- {OUTPUT_DIR / 'rys_4_10b_spektrogramy_augmentacji.png'}")
    logging.info("=== ZAKOŃCZONO ===")


if __name__ == "__main__":
    main()