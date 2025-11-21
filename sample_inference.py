"""Sample inference output for README"""
from train import infer_from_csv
import numpy as np

# Calm file
print("CALM FILE (csvs/calm_1.csv):")
labels, probs = infer_from_csv('ecg_pf_norm_30.keras', 'csvs/calm_1.csv')
print(f"  Chunks: {len(probs)}")
print(f"  Probabilities (first 5): {[f'{p:.3f}' for p in probs[:5]]}")
print(f"  Mean probability: {np.mean(probs):.4f}")
print(f"  Min/Max: {np.min(probs):.4f} / {np.max(probs):.4f}")
pred_label = "excited" if np.mean(probs) >= 0.5 else "calm"
print(f"  File-level prediction: {pred_label} (mean prob >= 0.5)")
print()

# Excited file  
print("EXCITED FILE (csvs/excited_1.csv):")
labels, probs = infer_from_csv('ecg_pf_norm_30.keras', 'csvs/excited_1.csv')
print(f"  Chunks: {len(probs)}")
print(f"  Probabilities (first 5): {[f'{p:.3f}' for p in probs[:5]]}")
print(f"  Mean probability: {np.mean(probs):.4f}")
print(f"  Min/Max: {np.min(probs):.4f} / {np.max(probs):.4f}")
pred_label = "excited" if np.mean(probs) >= 0.5 else "calm"
print(f"  File-level prediction: {pred_label} (mean prob >= 0.5)")
