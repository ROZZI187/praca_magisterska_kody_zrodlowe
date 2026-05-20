
Repozytorium zawiera skrypty w języku Python wykorzystane w pracy magisterskiej dotyczącej klasyfikacji dźwięków ptaków z użyciem metod uczenia maszynowego i głębokiego uczenia.

# Autor: Mateusz Urbański Politechnika Świętokrzyska

## Skrypty
### Przygotowanie danych

| Plik | Opis |
|---|---|
| `clean_annotations_dataset.py` | Oczyszczanie pliku adnotacji poprzez usunięcie rekordów z nieznanym gatunkiem. |
| `audio_preprocessing_pipeline.py` | Ekstrakcja segmentów audio, normalizacja sygnału, ujednolicenie długości nagrań oraz zapis metadanych. |
| `class_distribution_analysis.py` | Analiza liczebności klas oraz generowanie raportów dotyczących rozkładu danych. |
| `dataset_filtering_and_splitting.py` | Filtrowanie rzadkich klas oraz podział zbioru danych na zbiory treningowe, walidacyjne i testowe. |

### Analiza i wizualizacja sygnału audio

| Plik | Opis |
|---|---|
| `waveform_and_spectrogram.py` | Generowanie przebiegu czasowego sygnału oraz spektrogramu STFT. |
| `stft_window_comparison.py` | Porównanie działania STFT dla krótkiego i długiego okna analizy. |
| `spectrogram_vs_mel_comparison.py` | Porównanie klasycznego spektrogramu i Mel-spektrogramu. |
| `mel_filterbank_visualization.py` | Wizualizacja banku filtrów Mel. |
| `mfcc_feature_visualization.py` | Wizualizacja współczynników MFCC. |
| `acoustic_feature_analysis.py` | Analiza wybranych cech akustycznych, takich jak centroid widmowy, bandwidth, roll-off i ZCR. |
| `audio_sample_selection.py` | Wybór reprezentatywnych próbek audio do wizualizacji i rysunków. |
| `audio_augmentation_visualization.py` | Wizualizacja wpływu augmentacji danych audio na przebieg sygnału i spektrogram. |

### Trenowanie modeli

| Plik | Opis |
|---|---|
| `svm_classifier.py` | Trenowanie i ewaluacja modelu SVM. |
| `random_forest_classifier.py` | Trenowanie i ewaluacja modelu Random Forest. |
| `knn_classifier.py` | Trenowanie i ewaluacja modelu k-Nearest Neighbors. |
| `cnn_classifier.py` | Trenowanie i ewaluacja konwolucyjnej sieci neuronowej (CNN). |
| `lstm_classifier.py` | Trenowanie i ewaluacja modelu LSTM. |

### Analiza wyników

| Plik | Opis |
|---|---|
| `model_performance_comparison.py` | Porównanie wyników modeli oraz generowanie tabel i wykresów metryk. |
| `model_evaluation_and_visualization.py` | Generowanie końcowych tabel wyników, heatmap, wykresów oraz macierzy pomyłek. |

## Zastosowane modele

W pracy wykorzystano następujące modele:

- Support Vector Machine (SVM)
- Random Forest
- k-Nearest Neighbors (kNN)
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)

## Generowane wyniki

Skrypty generują między innymi:

- oczyszczone pliki adnotacji,
- przetworzone segmenty audio,
- raporty liczebności klas,
- wykresy i wizualizacje cech akustycznych,
- wytrenowane modele,
- raporty klasyfikacji,
- macierze pomyłek,
- końcowe tabele i wykresy porównawcze.

## Wykorzystane biblioteki

- Python
- NumPy
- pandas
- librosa
- matplotlib
- scikit-learn
- TensorFlow / Keras
