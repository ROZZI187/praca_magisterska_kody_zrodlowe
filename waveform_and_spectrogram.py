import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_path = "pigeon-fx.wav"
y, sr = librosa.load(audio_path, sr=None)

# przebieg czasowy
time = np.linspace(0, len(y)/sr, num=len(y))

# STFT
n_fft = 2048
hop_length = 256

S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
S_db = librosa.amplitude_to_db(S, ref=np.max)

plt.figure(figsize=(12,6))

# przebieg czasowy
plt.subplot(2,1,1)
plt.plot(time, y)
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda")
plt.title("Przebieg czasowy sygnału dźwiękowego")

# spektrogram
plt.subplot(2,1,2)
librosa.display.specshow(
    S_db,
    sr=sr,
    hop_length=hop_length,
    x_axis="time",
    y_axis="hz",
    vmin=-60,
    vmax=0
)

plt.ylim(0,2000)
plt.colorbar(format="%+2.0f dB")
plt.title("Spektrogram STFT sygnału (0–2 kHz)")

plt.tight_layout()
plt.savefig("rysunek_spektrogram.png", dpi=300)
plt.show()