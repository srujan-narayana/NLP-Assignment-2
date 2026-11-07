import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import json
import pickle
import numpy as np
import pandas as pd

from config import RESULTS_DIR, CHECKPOINTS_DIR, FIGURES_DIR, RANDOM_SEED
from common.reproducibility import set_seed
from common.visualization import plot_task1_vs_task3_comparison
from task1.dataset import get_tokenized_corpus
from task3.embedding_extractor import extract_layer_representations
from task3.static_vectors import construct_static_embeddings_from_bert
from task3.evaluation import evaluate_bert_static_embeddings
from task3.structural_check import evaluate_bert_static_structural

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Task3")

def run_task3(selected_layer: int = None):
    set_seed(RANDOM_SEED)
    output_dir = RESULTS_DIR / "task3"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = CHECKPOINTS_DIR / "task3"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting Task 3: BERT-Derived Static Word Vectors ===")

    # Determine best layer from Task 2 if not provided
    if selected_layer is None:
        best_layer_file = CHECKPOINTS_DIR / "task2" / "best_layer_info.json"
        if best_layer_file.exists():
            with open(best_layer_file, "r", encoding="utf-8") as f:
                info = json.load(f)
                selected_layer = info["best_layer"]
                logger.info(f"Loaded best layer from Task 2: Layer {selected_layer}")
        else:
            selected_layer = 7
            logger.warning(f"Task 2 best layer file not found, defaulting to Layer {selected_layer}")

    # 1. Extract representations from GUM train
    sentences = get_tokenized_corpus("train")
    layer_reps = extract_layer_representations(sentences, selected_layer)

    # 2. Average representations per word type
    static_embeddings, freqs = construct_static_embeddings_from_bert(sentences, layer_reps)

    # Save static embeddings
    with open(ckpt_dir / f"bert_static_embeddings_layer_{selected_layer}.pkl", "wb") as f:
        pickle.dump(static_embeddings, f)
    with open(ckpt_dir / "word_frequencies.json", "w", encoding="utf-8") as f:
        json.dump(freqs, f)

    # 3. Evaluate Semantic Benchmarks
    eval_res = evaluate_bert_static_embeddings(static_embeddings)
    wordsim_res = eval_res["wordsim353"]
    analogy_res = eval_res["google_analogy"]

    # 4. Evaluate Structural Check
    structural_res = evaluate_bert_static_structural(static_embeddings, default_dim=768)

    # 5. Load Task 1 results for side-by-side comparison
    task1_file = RESULTS_DIR / "task1" / "task1_results.json"
    task1_metrics = {}
    if task1_file.exists():
        with open(task1_file, "r", encoding="utf-8") as f:
            t1_data = json.load(f)
            task1_metrics = {
                "wordsim353_spearman": t1_data["wordsim353"]["spearman_correlation"],
                "google_analogy_accuracy": t1_data["google_analogy"]["accuracy"],
                "mst_attachment_agreement": t1_data["structural_check"]["overall_attachment_agreement"]
            }

    task3_metrics = {
        "wordsim353_spearman": wordsim_res["spearman_correlation"],
        "google_analogy_accuracy": analogy_res["accuracy"],
        "mst_attachment_agreement": structural_res["overall_attachment_agreement"]
    }

    # Comparison DataFrame
    comparison_data = [
        {
            "Metric": "WordSim-353 (Spearman ρ)",
            "Task 1: PPMI + SVD (Static)": f"{task1_metrics.get('wordsim353_spearman', 0.0):.4f}",
            "Task 3: BERT-Derived Static": f"{task3_metrics['wordsim353_spearman']:.4f}",
            "Delta (Task 3 - Task 1)": f"{task3_metrics['wordsim353_spearman'] - task1_metrics.get('wordsim353_spearman', 0.0):+.4f}"
        },
        {
            "Metric": "Google Analogy Accuracy",
            "Task 1: PPMI + SVD (Static)": f"{task1_metrics.get('google_analogy_accuracy', 0.0):.4f}",
            "Task 3: BERT-Derived Static": f"{task3_metrics['google_analogy_accuracy']:.4f}",
            "Delta (Task 3 - Task 1)": f"{task3_metrics['google_analogy_accuracy'] - task1_metrics.get('google_analogy_accuracy', 0.0):+.4f}"
        },
        {
            "Metric": "MST Attachment Agreement (UUAS on 20 sents)",
            "Task 1: PPMI + SVD (Static)": f"{task1_metrics.get('mst_attachment_agreement', 0.0):.4f}",
            "Task 3: BERT-Derived Static": f"{task3_metrics['mst_attachment_agreement']:.4f}",
            "Delta (Task 3 - Task 1)": f"{task3_metrics['mst_attachment_agreement'] - task1_metrics.get('mst_attachment_agreement', 0.0):+.4f}"
        }
    ]
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / "comparison_table.csv", index=False)
    logger.info("\n" + comparison_df.to_string(index=False))

    # Generate comparison plot
    plot_task1_vs_task3_comparison(task1_metrics, task3_metrics, FIGURES_DIR / "task1_vs_task3_comparison.png")

    summary = {
        "task": "Task 3: BERT-Derived Static Vectors",
        "selected_layer": selected_layer,
        "vocabulary_size": len(static_embeddings),
        "wordsim353": wordsim_res,
        "google_analogy": {
            "accuracy": analogy_res["accuracy"],
            "correct": analogy_res["correct"],
            "evaluated": analogy_res["evaluated"],
            "dropped": analogy_res["dropped"],
            "total_questions": analogy_res["total_questions"]
        },
        "structural_check": {
            "overall_attachment_agreement": structural_res["overall_attachment_agreement"],
            "macro_attachment_agreement": structural_res["macro_attachment_agreement"],
            "total_correct_edges": structural_res["total_correct_edges"],
            "total_gold_edges": structural_res["total_gold_edges"],
            "num_sentences": structural_res["num_sentences"]
        },
        "comparison": comparison_data
    }

    with open(output_dir / "task3_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== Task 3 Completed Successfully! ===")
    return summary

if __name__ == "__main__":
    run_task3()
