from pathlib import Path
import numpy as np
import librosa

BASE_DIR = Path(r"C:\Users\Mateusz-PC\Desktop\Praca magisterska\data_splits\train")

wav_files = sorted(BASE_DIR.rglob("*.wav"))
if not wav_files:
    raise FileNotFoundError(f"Nie znaleziono plików WAV w: {BASE_DIR}")

results = []

for path in wav_files[:2000]:  # limit, żeby nie mielić całego zbioru godzinami
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        y_trim, _ = librosa.effects.trim(y, top_db=25)

        if len(y_trim) < int(0.4 * sr):
            continue

        duration = len(y_trim) / sr
        rms = float(np.sqrt(np.mean(y_trim**2)))

        # prosta miara "treściwości" sygnału
        score = duration * rms

        results.append((score, duration, rms, str(path)))
    except Exception:
        continue

results = sorted(results, reverse=True)

print("TOP 20 kandydatów do rysunków:\n")
for i, (score, duration, rms, path) in enumerate(results[:20], 1):
    print(f"{i:02d}. score={score:.4f} | dur={duration:.3f}s | rms={rms:.4f} | {path}")
