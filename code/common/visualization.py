import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def plot_task2_layerwise_metrics(
    uuas_by_rank: Dict[int, List[float]],
    depth_corr_by_rank: Dict[int, List[float]],
    layers: List[int],
    save_path_prefix: Path
) -> None:
    """
    Generate Plot 1 (UUAS vs BERT layer) and Plot 2 (Depth Spearman vs BERT layer).
    """
    # 1. UUAS vs Layer
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {32: '#1f77b4', 64: '#ff7f0e', 128: '#2ca02c'}
    markers = {32: 'o', 64: 's', 128: '^'}

    for rank, scores in uuas_by_rank.items():
        ax.plot(layers, scores, label=f'Rank k={rank}', color=colors.get(rank, 'blue'),
                marker=markers.get(rank, 'o'), linewidth=2, markersize=7)

    ax.set_xlabel('BERT Layer')
    ax.set_ylabel('UUAS (Undirected Attachment Score)')
    ax.set_title('Task 2: Dependency Tree UUAS vs. BERT Layer')
    ax.set_xticks(layers)
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_uuas_vs_layer.png")
    plt.close()

    # 2. Depth Correlation vs Layer
    fig, ax = plt.subplots(figsize=(8, 5))
    for rank, scores in depth_corr_by_rank.items():
        ax.plot(layers, scores, label=f'Rank k={rank}', color=colors.get(rank, 'blue'),
                marker=markers.get(rank, 'o'), linewidth=2, markersize=7)

    ax.set_xlabel('BERT Layer')
    ax.set_ylabel("Spearman's Rank Correlation (ρ)")
    ax.set_title('Task 2: Depth Probe Spearman Correlation vs. BERT Layer')
    ax.set_xticks(layers)
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_depth_corr_vs_layer.png")
    plt.close()

def plot_task1_vs_task3_comparison(
    task1_metrics: Dict[str, float],
    task3_metrics: Dict[str, float],
    save_path: Path
) -> None:
    """
    Generate side-by-side comparison bar chart for Task 1 vs Task 3.
    """
    metrics = ['WordSim-353 (ρ)', 'Google Analogy Acc', 'MST Attachment Acc']
    t1_vals = [
        task1_metrics.get('wordsim353_spearman', 0.0),
        task1_metrics.get('google_analogy_accuracy', 0.0),
        task1_metrics.get('mst_attachment_agreement', 0.0)
    ]
    t3_vals = [
        task3_metrics.get('wordsim353_spearman', 0.0),
        task3_metrics.get('google_analogy_accuracy', 0.0),
        task3_metrics.get('mst_attachment_agreement', 0.0)
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, t1_vals, width, label='Task 1: PPMI + SVD (Static)', color='#4c72b0')
    rects2 = ax.bar(x + width/2, t3_vals, width, label='Task 3: BERT-Derived Static', color='#dd8452')

    ax.set_ylabel('Score / Accuracy')
    ax.set_title('Comparison: Classical Static (PPMI+SVD) vs. Contextual-Derived Static (BERT)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim([0.0, 1.0])
    ax.legend(frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
