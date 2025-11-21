"""
Generate inference plots for sample ECG files.
Saves plots to plots/ directory.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from train import infer_from_csv

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

model_path = 'ecg_pf_norm_30.keras'

# Sample files to run inference on
sample_files = [
    'csvs/calm_1.csv',
    'csvs/excited_1.csv',
]

for csv_path in sample_files:
    if not os.path.exists(csv_path):
        print(f"[SKIP] {csv_path} not found")
        continue
    
    print(f"\n[INFERENCE] Processing {csv_path}...")
    
    try:
        labels, probs = infer_from_csv(model_path, csv_path)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Plot 1: Per-chunk probabilities over time
        chunks = np.arange(len(probs))
        ax1.plot(chunks, probs, 'o-', linewidth=2, markersize=6, color='steelblue', label='P(excited)')
        ax1.axhline(0.5, color='red', linestyle='--', linewidth=2.5, label='Decision threshold (0.5)')
        ax1.fill_between(chunks, 0, probs, where=(probs >= 0.5), alpha=0.3, color='red', label='Predicted excited')
        ax1.fill_between(chunks, 0, probs, where=(probs < 0.5), alpha=0.3, color='blue', label='Predicted calm')
        ax1.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Chunk index', fontsize=12, fontweight='bold')
        ax1.set_title(f'Per-Chunk Predictions: {os.path.basename(csv_path)}', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=10, loc='upper right')
        ax1.set_ylim([-0.05, 1.05])
        
        # Plot 2: Probability distribution histogram
        ax2.hist(probs, bins=max(10, len(probs)//3), alpha=0.7, edgecolor='black', color='steelblue', label='Chunk probs')
        mean_prob = np.mean(probs)
        ax2.axvline(mean_prob, color='green', linestyle='-', linewidth=2.5, label=f'Mean: {mean_prob:.3f}')
        ax2.axvline(0.5, color='red', linestyle='--', linewidth=2.5, label='Threshold: 0.5')
        ax2.set_xlabel('Probability', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Probability Distribution Across All Chunks', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10, loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        filename = os.path.basename(csv_path).replace('.csv', '_inference.png')
        output_path = os.path.join('plots', filename)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"[SAVED] Plot to {output_path}")
        
        # Compute file-level prediction
        file_pred = 'excited' if mean_prob >= 0.5 else 'calm'
        n_excited = np.sum(labels)
        n_calm = len(labels) - n_excited
        
        # Print summary
        print(f"  Chunks: {len(probs)} | Mean prob: {mean_prob:.4f} | File prediction: {file_pred}")
        print(f"  Per-chunk votes: calm={n_calm}, excited={n_excited}")
        
        plt.close(fig)
    
    except Exception as e:
        print(f"[ERROR] {csv_path}: {e}")

print("\n[COMPLETE] Inference plots saved to plots/")
