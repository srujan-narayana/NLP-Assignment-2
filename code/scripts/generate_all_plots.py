import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def generate_plots():
    # Load results
    with open(RESULTS_DIR / "task1" / "task1_results.json", "r") as f:
        t1 = json.load(f)
    with open(RESULTS_DIR / "task2" / "task2_results.json", "r") as f:
        t2 = json.load(f)
    with open(RESULTS_DIR / "task3" / "task3_results.json", "r") as f:
        t3 = json.load(f)

    # 1. Task 1 Summary Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    metrics_t1 = ['WordSim-353 (ρ)', 'Google Analogy Acc', 'MST Agreement (UUAS)']
    vals_t1 = [
        t1["wordsim353"]["spearman_correlation"],
        t1["google_analogy"]["accuracy"],
        t1["structural_check"]["overall_attachment_agreement"]
    ]
    colors = ['#4c72b0', '#55a868', '#c44e52']
    bars = ax.bar(metrics_t1, vals_t1, color=colors, width=0.45)
    ax.set_ylim([0.0, 0.35])
    ax.set_ylabel('Score / Correlation / Accuracy')
    ax.set_title('Task 1: PPMI + SVD Static Embeddings Evaluation Summary')
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.4f}', xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task1_summary.png")
    plt.close()

    # 2. Task 2 Rank Comparison Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ranks = [32, 64, 128]
    best_uuas_by_rank = []
    for r in ranks:
        # results_by_rank keys might be str or int in json
        r_str = str(r)
        r_dict = t2["results_by_rank"].get(r_str, t2["results_by_rank"].get(r, {}))
        max_u = max(v["test_uuas"] for v in r_dict.values())
        best_uuas_by_rank.append(max_u)

    bars = ax.bar([f'Rank k={r}' for r in ranks], best_uuas_by_rank, color=['#1f77b4', '#aec7e8', '#ff7f0e'], width=0.45)
    ax.set_ylim([0.0, 0.70])
    ax.set_ylabel('Peak Test UUAS')
    ax.set_title('Task 2: Effect of Probe Rank k on Peak Syntactic UUAS')
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.4f}', xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task2_rank_comparison.png")
    plt.close()

    # 3. Overall 3-way Synthesis Plot (Task 1 vs Task 2 vs Task 3 on Syntax vs Semantics)
    fig, ax = plt.subplots(figsize=(9, 5))
    categories = ['Semantic Sim\n(WordSim-353 ρ)', 'Relational Analogy\n(Google Analogy Acc)', 'Syntactic Structure\n(Test Tree UUAS)']
    t1_scores = [t1["wordsim353"]["spearman_correlation"], t1["google_analogy"]["accuracy"], t1["structural_check"]["overall_attachment_agreement"]]
    t2_scores = [0.0, 0.0, t2["results_by_rank"]["64"]["8"]["test_uuas"]]  # Task 2 contextual probe
    t3_scores = [t3["wordsim353"]["spearman_correlation"], t3["google_analogy"]["accuracy"], t3["structural_check"]["overall_attachment_agreement"]]

    x = np.arange(len(categories))
    w = 0.25

    ax.bar(x - w, t1_scores, width=w, label='Task 1: Classical PPMI+SVD (Static)', color='#4c72b0')
    ax.bar(x, t2_scores, width=w, label='Task 2: BERT Contextual Probe (L8, k=64)', color='#2ca02c')
    ax.bar(x + w, t3_scores, width=w, label='Task 3: BERT-Derived Static', color='#dd8452')

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Score / Accuracy / UUAS')
    ax.set_title('Cross-Task Evaluation: Static Distributional vs. Contextual vs. Pooled Static')
    ax.set_ylim([0.0, 0.75])
    ax.legend(frameon=True, loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "all_tasks_comparison.png")
    plt.close()
    print("All supplementary plots successfully generated in", FIGURES_DIR)

if __name__ == "__main__":
    generate_plots()
