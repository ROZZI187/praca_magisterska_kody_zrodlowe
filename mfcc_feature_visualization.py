from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_path = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\data_splits\train\aldfly\SSW_250_seg0015_aldfly.wav")
out_dir = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\results\figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "rys_5_4_mfcc.png"

y, sr = librosa.load(audio_path, sr=None, mono=True)

print(f"Użyty plik: {audio_path}")
print(f"Częstotliwość próbkowania: {sr} Hz")
print(f"Długość oryginalna: {len(y)/sr:.3f} s")

y_trim, _ = librosa.effects.trim(y, top_db=25)
print(f"Długość po trim: {len(y_trim)/sr:.3f} s")

max_duration = 1.5
max_samples = int(max_duration * sr)

if len(y_trim) > max_samples:
    y_plot = y_trim[:max_samples]
else:
    y_plot = y_trim

plot_duration = len(y_plot) / sr
print(f"Długość fragmentu na rysunku: {plot_duration:.3f} s")

n_fft = 1024
hop_length = 256
n_mfcc = 13
n_mels = 64

mfcc = librosa.feature.mfcc(
    y=y_plot,
    sr=sr,
    n_mfcc=n_mfcc,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels
)

mfcc_mean = np.mean(mfcc, axis=1, keepdims=True)
mfcc_std = np.std(mfcc, axis=1, keepdims=True) + 1e-8
mfcc_norm = (mfcc - mfcc_mean) / mfcc_std

plt.figure(figsize=(12, 6))

img = librosa.display.specshow(
    mfcc_norm,
    x_axis="time",
    sr=sr,
    hop_length=hop_length,
    cmap="magma"
)

plt.title("Współczynniki MFCC")
plt.xlabel("Czas [s]")
plt.ylabel("Numer współczynnika MFCC")

cbar = plt.colorbar(img, format="%+2.1f")
cbar.set_label("Wartość znormalizowana")

plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Zapisano rysunek: {out_path}")