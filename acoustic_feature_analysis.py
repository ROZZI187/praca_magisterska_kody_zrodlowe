from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_path = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\data_splits\train\aldfly\SSW_250_seg0015_aldfly.wav")
out_dir = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\results\figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "rys_5_5_cechy_akustyczne.png"

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

spectral_centroid = librosa.feature.spectral_centroid(
    y=y_plot, sr=sr, n_fft=n_fft, hop_length=hop_length
)[0]

spectral_bandwidth = librosa.feature.spectral_bandwidth(
    y=y_plot, sr=sr, n_fft=n_fft, hop_length=hop_length
)[0]

spectral_rolloff = librosa.feature.spectral_rolloff(
    y=y_plot, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
)[0]

zcr = librosa.feature.zero_crossing_rate(
    y_plot, hop_length=hop_length
)[0]

frames = np.arange(len(spectral_centroid))
times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

axes[0].plot(times, spectral_centroid, linewidth=1.5)
axes[0].set_title("Centroid widmowy")
axes[0].set_ylabel("Hz")
axes[0].grid(alpha=0.25)

axes[1].plot(times, spectral_bandwidth, linewidth=1.5)
axes[1].set_title("Szerokość pasma")
axes[1].set_ylabel("Hz")
axes[1].grid(alpha=0.25)

axes[2].plot(times, spectral_rolloff, linewidth=1.5)
axes[2].set_title("Roll-off widmowy")
axes[2].set_ylabel("Hz")
axes[2].grid(alpha=0.25)

axes[3].plot(times, zcr, linewidth=1.5)
axes[3].set_title("Zero-crossing rate")
axes[3].set_ylabel("ZCR")
axes[3].set_xlabel("Czas [s]")
axes[3].grid(alpha=0.25)

for ax in axes:
    ax.set_xlim(0, plot_duration)

plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Zapisano rysunek: {out_path}")