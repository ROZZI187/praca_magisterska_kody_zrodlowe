from pathlib import Path
import librosa
import librosa.filters
import matplotlib.pyplot as plt
import numpy as np

sr = 22050
n_fft = 1024
n_mels = 20

out_dir = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\results\figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "rys_5_3_bank_filtrow_mel_heatmap.png"

mel_basis = librosa.filters.mel(
    sr=sr,
    n_fft=n_fft,
    n_mels=n_mels,
    fmin=0,
    fmax=8000
)

plt.figure(figsize=(10, 6))

mel_log = np.log(mel_basis + 1e-6)

plt.imshow(
    mel_log,
    aspect='auto',
    origin='lower',
    extent=[0, 8000, 0, n_mels]
)

plt.colorbar(label="Wzmocnienie")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Numer filtru Mel")
plt.title("Rozmieszczenie filtrów w skali Mel")

plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Zapisano rysunek: {out_path}")