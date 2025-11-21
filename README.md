# ECG Binary Classifier: Calm vs. Excited

A TensorFlow/Keras deep learning pipeline for classifying ECG signals into two states: **calm** and **excited**. The model uses Conv1D blocks combined with engineered time-domain and spectral features for robust discrimination.

## Overview

- **Task:** Binary ECG classification (calm vs. excited)
- **Architecture:** Two-input model — Conv1D branch + engineered feature branch, merged and passed through dense layers with sigmoid output
- **Preprocessing:** Per-file z-score normalization, non-overlapping 1000-sample chunking
- **Features:** 16-dimensional per-chunk feature vector (time-domain, FFT bands, spectral metrics)
- **Final Metrics (30 epochs):** Test Accuracy 0.83 | ROC-AUC 0.91 | PR-AUC 0.96
- **Model Formats:** Native Keras `.keras` (recommended) + legacy HDF5 `.h5`

## Replicability Guide

### 1. Environment Setup

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install tensorflow numpy pandas scikit-learn scipy matplotlib
```

### 2. Data Format

Place CSV files in `csvs/` folder with naming convention:
- `calm_*.csv` — calm state examples (5 files)
- `excited_*.csv` — excited state examples (12 files)


**CSV Structure:**
- One or more numeric columns; first numeric column (or voltage-like) used as ECG signal
- `time` columns excluded; NaN values dropped before processing

### 3. Training

```powershell
# Full training (30 epochs, recommended)
python train.py \
  --csv_dir csvs \
  --chunk_size 1000 \
  --epochs 30 \
  --batch_size 8 \
  --save ecg_model_final.keras

# Quick validation (1 epoch)
python train.py --csv_dir csvs --epochs 1 --batch_size 8 --save ecg_test.keras

# Overfit mode (verify model capacity)
python train.py --csv_dir csvs --epochs 30 --batch_size 8 --overfit
```

### 4. Model Parameters & Reproducibility

- **Train/Test Split:** 80/20 stratified split
- **Optimizer:** Adam lr=1e-4, gradient clipping (clipnorm=1.0)
- **Loss:** Binary crossentropy
- **Metrics:** Accuracy, AUC (ROC), PR-AUC (average precision)
- **Regularization:** BatchNorm, L2 regularization

### 5. Preprocessing Details

1. **Per-file normalization:** Z-score each file independently before chunking
   - Formula: `(signal - mean) / std`
   - Preserves intra-file amplitude variations

2. **Chunking:** Non-overlapping 1000-sample chunks from normalized signals

3. **Feature extraction (16-dim):**
   - **Time-domain (7):** mean, std, median, min, max, ptp, RMS
   - **FFT bands (3):** low-energy (0–50 Hz), mid-energy (50–100 Hz), high-energy (100+ Hz)
   - **Spectral (6):** skewness, kurtosis, spectral centroid, entropy, dominant frequency, flatness

### 6. Inference

**Python API:**
```python
from train import infer_from_csv
import numpy as np

labels, probs = infer_from_csv('ecg_pf_norm_30.keras', 'csvs/calm_1.csv')
print(f"Per-chunk probabilities: {probs}")
print(f"File-level prediction: {'excited' if np.mean(probs) >= 0.5 else 'calm'}")
```

**Inference returns:**
- `labels`: chunk-level binary predictions (0=calm, 1=excited) — list of strings
- `probs`: positive class probability (P(excited)) for each chunk — 1D numpy array [0, 1]

**File-level classification:** Average probabilities across chunks and threshold at 0.5:
```python
file_pred = "excited" if np.mean(probs) >= 0.5 else "calm"
```

#### Sample Inference Output

**CALM FILE (csvs/calm_1.csv):**
- Chunks: 59 | Mean probability: 0.1667 | Range: [0.0000, 0.9806]
- Probabilities: [0.000, 0.001, 0.009, 0.036, 0.030, ...]
- **File-level prediction: calm** (mean prob = 0.1667 < 0.5)

**EXCITED FILE (csvs/excited_1.csv):**
- Chunks: 60 | Mean probability: 0.8350 | Range: [0.0532, 1.0000]
- Probabilities: [0.991, 0.972, 0.510, 0.461, 0.491, ...]
- **File-level prediction: excited** (mean prob = 0.8350 >= 0.5)

### 7. Visualization & Inference Plots

Inference plots (per-chunk probabilities + distribution) are saved in `plots/`:

```powershell
# Generate plots for sample files
python generate_inference_plots.py
```

**Output files:**
- `plots/calm_1_inference.png` — Calm file predictions
- `plots/excited_1_inference.png` — Excited file predictions

Each plot shows:
1. **Top panel:** Per-chunk probabilities over time with decision threshold (red dashed line at 0.5)
   - Blue shading: predicted calm (prob < 0.5)
   - Red shading: predicted excited (prob >= 0.5)
2. **Bottom panel:** Histogram of probability distribution
   - Green line: mean probability across chunks
   - Red dashed line: decision threshold (0.5)

## Final Training Results (30 Epochs, Per-File Normalization)

**Test Set (183 samples):**
- Accuracy: 0.8251
- ROC-AUC: 0.9052
- PR-AUC: 0.9621

**Classification Report:**
```
           precision  recall  f1-score  support
calm            0.73    0.62      0.67       52
excited         0.86    0.91      0.88      131
accuracy                          0.83      183
```

**Confusion Matrix:**
```
[[32  20]    TN=32, FP=20
 [12 119]]   FN=12, TP=119
```

## Model Files

- **`ecg_pf_norm_30.keras`** — Native Keras format (recommended, TensorFlow 2.13+)
- **`ecg_pf_norm_30.h5`** — HDF5 legacy format (still functional)

## Repository Structure

```
ECG/
├── csvs/                              # Input ECG CSV files
│   ├── calm_*.csv                     # Calm state recordings (5 files)       
│   └── excited*.csv                    # Excited state recordings (12 files)
├── plots/                             # Output inference visualization plots
│   ├── calm_1_inference.png
│   └── excited_1_inference.png
├── train.py                           # Main training + inference pipeline
├── generate_inference_plots.py         # Generate visualization plots
├── sample_inference.py                 # Example inference output
├── ecg_pf_norm_30.keras              # Final trained model (Keras format)
├── ecg_pf_norm_30.h5                 # Final trained model (HDF5 legacy)
└── README.md                          # This file
```

## Code Architecture

- **`load_csv_signal(path)`** — CSV loader with numeric column selection, NaN dropping
- **`make_chunks(signal, chunk_size)`** — Non-overlapping chunking
- **`extract_features_from_chunk(chunk)`** — 16-d feature computation
- **`build_dataset(csv_dir, chunk_size)`** — Full pipeline: load → normalize → chunk → extract
- **`build_conv1d_model(input_length)`** — Two-input model (raw + features)
- **`train_and_evaluate(...)`** — Training with test evaluation
- **`infer_from_csv(model_path, csv_path)`** — Per-chunk predictions with probabilities

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model not found" | Use absolute path or ensure `.keras`/`.h5` in project root |
| NaN loss during training | Check for inf values in data; current settings (lr=1e-4, clipnorm=1.0) are stable |
| Imbalanced predictions | Calm recall (0.62) lower due to class imbalance; try focal loss or oversampling |
| Non-reproducible results | Set seeds: `tf.random.set_seed(42); np.random.seed(42)` |

## Future Improvements

- R-peak detection + HRV features (SDNN, RMSSD, pNN50)
- Class-balanced sampling or focal loss
- EarlyStopping with best validation checkpoint
- Attention mechanisms for interpretability
- ONNX/TFLite export for deployment


