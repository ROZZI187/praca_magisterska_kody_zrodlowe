from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

BASE_DIR = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\data_splits")
OUT_DIR = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\results\figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "rys_5_1_stft_okna.png"

wav_files = sorted(BASE_DIR.rglob("*.wav"))
if not wav_files:
    raise FileNotFoundError(f"Nie znaleziono plików WAV w: {BASE_DIR}")

audio_path = wav_files[0]
print("Użyty plik:", audio_path)

y, sr = librosa.load(audio_path, sr=None, mono=True)

print(f"Częstotliwość próbkowania: {sr} Hz")
print(f"Długość oryginalna: {len(y)/sr:.3f} s")

y_trim, idx = librosa.effects.trim(y, top_db=25)

print(f"Długość po trim: {len(y_trim)/sr:.3f} s")

max_duration = 0.6  # sekundy
max_samples = int(max_duration * sr)

if len(y_trim) > max_samples:
    y_plot = y_trim[:max_samples]
else:
    y_plot = y_trim

plot_duration = len(y_plot) / sr
print(f"Długość fragmentu na rysunku: {plot_duration:.3f} s")

# Krótkie okno: lepsza rozdzielczość czasowa
n_fft_short = 512
hop_short = 128

# Długie okno: lepsza rozdzielczość częstotliwościowa
n_fft_long = 2048
hop_long = 512

D_short = librosa.stft(y_plot, n_fft=n_fft_short, hop_length=hop_short, window="hann")
S_short_db = librosa.amplitude_to_db(np.abs(D_short), ref=np.max)

D_long = librosa.stft(y_plot, n_fft=n_fft_long, hop_length=hop_long, window="hann")
S_long_db = librosa.amplitude_to_db(np.abs(D_long), ref=np.max)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

# GÓRNY WYKRES - krótkie okno
img1 = librosa.display.specshow(
    S_short_db,
    sr=sr,
    hop_length=hop_short,
    x_axis="time",
    y_axis="hz",
    ax=axes[0]
)
axes[0].set_title("STFT – krótkie okno analizy")
axes[0].set_xlabel("Czas [s]")
axes[0].set_ylabel("Częstotliwość [Hz]")
axes[0].set_xlim(0, plot_duration)
axes[0].set_ylim(0, 6000)

# DOLNY WYKRES - długie okno
img2 = librosa.display.specshow(
    S_long_db,
    sr=sr,
    hop_length=hop_long,
    x_axis="time",
    y_axis="hz",
    ax=axes[1]
)
axes[1].set_title("STFT – długie okno analizy")
axes[1].set_xlabel("Czas [s]")
axes[1].set_ylabel("Częstotliwość [Hz]")
axes[1].set_xlim(0, plot_duration)
axes[1].set_ylim(0, 6000)

cbar = fig.colorbar(img2, ax=axes, format="%+2.0f dB", shrink=0.95)
cbar.set_label("Poziom [dB]")

plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"Zapisano rysunek: {OUT_PATH}")