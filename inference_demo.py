"""
Inference demonstration: run model on sample CSV and show per-chunk probabilities
"""
from train import infer_from_csv
import numpy as np
import matplotlib.pyplot as plt

def demo_inference(csv_path, model_path='ecg_pf_norm_30.keras'):
    """Run inference and show results"""
    print(f'Running inference on: {csv_path}')
    labels, probs = infer_from_csv(model_path, csv_path)
    
    # Flatten probs if 2D
    probs = np.array(probs).ravel()
    
    print(f'\n=== INFERENCE RESULTS ===')
    print(f'Number of chunks: {len(probs)}')
    print(f'Per-chunk predictions (first 10): {probs[:10]}')
    print(f'Per-chunk predictions (last 10): {probs[-10:]}')
    print(f'\nMean probability: {np.mean(probs):.4f}')
    print(f'Std probability: {np.std(probs):.4f}')
    print(f'Min probability: {np.min(probs):.4f}')
    print(f'Max probability: {np.max(probs):.4f}')
    
    pred = 'EXCITED' if np.mean(probs) > 0.5 else 'CALM'
    confidence = abs(np.mean(probs) - 0.5) * 2 * 100
    print(f'\n=== FILE-LEVEL PREDICTION ===')
    print(f'Predicted class: {pred}')
    print(f'Confidence: {confidence:.1f}%')
    
    print(f'\n=== PROBABILITY DISTRIBUTION ===')
    strong_calm = sum(probs < 0.3)
    ambiguous = sum((probs >= 0.3) & (probs < 0.7))
    strong_excited = sum(probs >= 0.7)
    print(f'Chunks with P(excited) < 0.3 (strong calm): {strong_calm} / {len(probs)} ({100*strong_calm/len(probs):.1f}%)')
    print(f'Chunks with 0.3 <= P(excited) < 0.7 (ambiguous): {ambiguous} / {len(probs)} ({100*ambiguous/len(probs):.1f}%)')
    print(f'Chunks with P(excited) >= 0.7 (strong excited): {strong_excited} / {len(probs)} ({100*strong_excited/len(probs):.1f}%)')
    
    return probs, labels

if __name__ == '__main__':
    # Run on calm example
    print('='*60)
    print('EXAMPLE 1: CALM STATE (csvs/calm_1.csv)')
    print('='*60)
    calm_probs, calm_labels = demo_inference('csvs/calm_1.csv')
    
    # Run on excited example
    print('\n' + '='*60)
    print('EXAMPLE 2: EXCITED STATE (csvs/excited_1.csv)')
    print('='*60)
    excited_probs, excited_labels = demo_inference('csvs/excited_1.csv')
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Plot 1: Calm per-chunk probs
    ax = axes[0, 0]
    ax.plot(calm_probs, 'o-', linewidth=1.5, markersize=4, color='blue', label='P(excited)')
    ax.axhline(0.5, color='r', linestyle='--', linewidth=2, label='Decision threshold')
    ax.fill_between(range(len(calm_probs)), 0, 0.3, alpha=0.2, color='blue', label='Strong calm')
    ax.fill_between(range(len(calm_probs)), 0.7, 1.0, alpha=0.2, color='red', label='Strong excited')
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title('CALM: Per-Chunk Predictions (csvs/calm_1.csv)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim([0, 1])
    
    # Plot 2: Calm histogram
    ax = axes[0, 1]
    ax.hist(calm_probs, bins=20, alpha=0.7, edgecolor='black', color='blue')
    ax.axvline(np.mean(calm_probs), color='g', linestyle='-', linewidth=2, label=f'Mean: {np.mean(calm_probs):.3f}')
    ax.axvline(0.5, color='r', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Probability', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('CALM: Probability Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    
    # Plot 3: Excited per-chunk probs
    ax = axes[1, 0]
    ax.plot(excited_probs, 'o-', linewidth=1.5, markersize=4, color='red', label='P(excited)')
    ax.axhline(0.5, color='r', linestyle='--', linewidth=2, label='Decision threshold')
    ax.fill_between(range(len(excited_probs)), 0, 0.3, alpha=0.2, color='blue', label='Strong calm')
    ax.fill_between(range(len(excited_probs)), 0.7, 1.0, alpha=0.2, color='red', label='Strong excited')
    ax.set_ylabel('Probability', fontsize=11)
    ax.set_title('EXCITED: Per-Chunk Predictions (csvs/excited_1.csv)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim([0, 1])
    ax.set_xlabel('Chunk index', fontsize=11)
    
    # Plot 4: Excited histogram
    ax = axes[1, 1]
    ax.hist(excited_probs, bins=20, alpha=0.7, edgecolor='black', color='red')
    ax.axvline(np.mean(excited_probs), color='g', linestyle='-', linewidth=2, label=f'Mean: {np.mean(excited_probs):.3f}')
    ax.axvline(0.5, color='r', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Probability', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('EXCITED: Probability Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('inference_plot.png', dpi=100, bbox_inches='tight')
    print('\nPlot saved to: inference_plot.png')
    plt.show()
