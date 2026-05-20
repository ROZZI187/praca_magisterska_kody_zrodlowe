from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

audio_path = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\data_splits\train\amecro\SSW_267_seg0093_amecro.wav")
out_dir = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\results\figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "rys_5_2_spektrogram_vs_mel.png"

y, sr = librosa.load(audio_path, sr=None, mono=True)

print(f"Użyty plik: {audio_path}")
print(f"Częstotliwość próbkowania: {sr} Hz")
print(f"Długość oryginalna: {len(y)/sr:.3f} s")

y_trim, idx = librosa.effects.trim(y, top_db=25)

print(f"Długość po trim: {len(y_trim)/sr:.3f} s")

max_duration = 1.5  # sekundy
max_samples = int(max_duration * sr)

if len(y_trim) > max_samples:
    y_plot = y_trim[:max_samples]
else:
    y_plot = y_trim

plot_duration = len(y_plot) / sr
print(f"Długość fragmentu na rysunku: {plot_duration:.3f} s")


n_fft = 1024
hop_length = 256
n_mels = 64  # celowo mniej niż 128, żeby różnica była bardziej czytelna

D = librosa.stft(y_plot, n_fft=n_fft, hop_length=hop_length, window="hann")
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
M = librosa.feature.melspectrogram(
    y=y_plot,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
    fmin=0,
    fmax=sr // 2,
    power=2.0
)
M_db = librosa.power_to_db(M, ref=np.max)
fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

img1 = librosa.display.specshow(
    S_db,
    sr=sr,
    hop_length=hop_length,
    x_axis="time",
    y_axis="hz",
    ax=axes[0]
)
axes[0].set_title("Spektrogram (skala liniowa)")
axes[0].set_xlabel("Czas [s]")
axes[0].set_ylabel("Częstotliwość [Hz]")
axes[0].set_xlim(0, plot_duration)
axes[0].set_ylim(0, 6000)
axes[0].tick_params(labelsize=10)

img2 = librosa.display.specshow(
    M_db,
    sr=sr,
    hop_length=hop_length,
    x_axis="time",
    y_axis="mel",
    ax=axes[1]
)
axes[1].set_title("Mel-spektrogram (skala Mel)")
axes[1].set_xlabel("Czas [s]")
axes[1].set_ylabel("Częstotliwość [Mel]")
axes[1].set_xlim(0, plot_duration)
axes[1].tick_params(labelsize=10)

cbar = fig.colorbar(img2, ax=axes, format="%+2.0f dB", shrink=0.95)
cbar.set_label("Poziom [dB]")

plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Zapisano rysunek: {out_path}")