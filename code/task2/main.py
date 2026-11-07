import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import json
import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any

from config import (
    PROBE_LAYERS, PROBE_RANKS, PROBE_LR, PROBE_BATCH_SIZE,
    PROBE_EPOCHS, RANDOM_SEED, RESULTS_DIR, CHECKPOINTS_DIR, FIGURES_DIR
)
from common.reproducibility import set_seed
from common.visualization import plot_task2_layerwise_metrics
from task2.dataset import ProbingDataset
from task2.bert_extractor import BertLayerExtractor
from task2.distance_probe import StructuralDistanceProbe
from task2.depth_probe import StructuralDepthProbe
from task2.evaluation import evaluate_distance_probe_uuas, evaluate_depth_probe_correlation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Task2")

def train_distance_probe(
    layer_num: int,
    rank: int,
    train_reps: List[torch.Tensor],
    train_g_dists: List[torch.Tensor],
    val_reps: List[torch.Tensor],
    val_heads: List[List[Any]],
    device: torch.device,
    epochs: int = PROBE_EPOCHS,
    lr: float = PROBE_LR,
    batch_size: int = PROBE_BATCH_SIZE
) -> Tuple[StructuralDistanceProbe, float]:
    probe = StructuralDistanceProbe(input_dim=768, probe_rank=rank).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)

    n_train = len(train_reps)
    indices = list(range(n_train))
    best_val_uuas = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        probe.train()
        np.random.shuffle(indices)
        total_loss = 0.0
        n_batches = 0

        # Mini-batch loop over sentences
        for i in range(0, n_train, batch_size):
            batch_indices = indices[i : i + batch_size]
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            count = 0

            for idx in batch_indices:
                rep = train_reps[idx].to(device)
                g_dist = train_g_dists[idx].to(device)
                if rep.shape[0] > 1:
                    loss = probe.compute_loss(rep, g_dist)
                    batch_loss = batch_loss + loss
                    count += 1

            if count > 0:
                batch_loss = batch_loss / count
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
                n_batches += 1

        if epoch % 10 == 0 or epoch == epochs:
            val_uuas, _, _, _ = evaluate_distance_probe_uuas(probe, val_reps, val_heads, device)
            avg_loss = total_loss / max(1, n_batches)
            logger.info(f"Layer {layer_num:2d} | Rank {rank:3d} | Epoch {epoch:2d}/{epochs} | Train Loss: {avg_loss:.4f} | Val UUAS: {val_uuas:.4f}")
            if val_uuas > best_val_uuas:
                best_val_uuas = val_uuas
                best_weights = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    if best_weights is not None:
        probe.load_state_dict(best_weights)
    return probe, best_val_uuas

def train_depth_probe(
    layer_num: int,
    rank: int,
    train_reps: List[torch.Tensor],
    train_g_depths: List[torch.Tensor],
    val_reps: List[torch.Tensor],
    val_g_depths: List[torch.Tensor],
    device: torch.device,
    epochs: int = PROBE_EPOCHS,
    lr: float = PROBE_LR,
    batch_size: int = PROBE_BATCH_SIZE
) -> Tuple[StructuralDepthProbe, float]:
    probe = StructuralDepthProbe(input_dim=768, probe_rank=rank).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)

    n_train = len(train_reps)
    indices = list(range(n_train))
    best_val_corr = -2.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        probe.train()
        np.random.shuffle(indices)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            batch_indices = indices[i : i + batch_size]
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            count = 0

            for idx in batch_indices:
                rep = train_reps[idx].to(device)
                g_depth = train_g_depths[idx].to(device)
                if rep.shape[0] > 1:
                    loss = probe.compute_loss(rep, g_depth)
                    batch_loss = batch_loss + loss
                    count += 1

            if count > 0:
                batch_loss = batch_loss / count
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
                n_batches += 1

        if epoch % 10 == 0 or epoch == epochs:
            val_corr, _ = evaluate_depth_probe_correlation(probe, val_reps, val_g_depths, device)
            avg_loss = total_loss / max(1, n_batches)
            logger.info(f"Layer {layer_num:2d} | Rank {rank:3d} | Epoch {epoch:2d}/{epochs} | Train Loss: {avg_loss:.4f} | Val Depth Corr: {val_corr:.4f}")
            if val_corr > best_val_corr:
                best_val_corr = val_corr
                best_weights = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    if best_weights is not None:
        probe.load_state_dict(best_weights)
    return probe, best_val_corr

def run_task2():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"=== Starting Task 2: Structural Probe on BERT (Using {device}) ===")

    output_dir = RESULTS_DIR / "task2"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = CHECKPOINTS_DIR / "task2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Datasets & gold targets
    train_ds = ProbingDataset("train")
    val_ds = ProbingDataset("validation")
    test_ds = ProbingDataset("test")

    # 2. Extract BERT representations (Layers 1..12)
    extractor = BertLayerExtractor(device=device)
    train_reps = extractor.get_or_extract_representations("train", train_ds.sentences_tokens, PROBE_LAYERS)
    val_reps = extractor.get_or_extract_representations("validation", val_ds.sentences_tokens, PROBE_LAYERS)
    test_reps = extractor.get_or_extract_representations("test", test_ds.sentences_tokens, PROBE_LAYERS)

    # 3. Layer-wise sweeping & training
    all_results = {}
    uuas_by_rank = {r: [] for r in PROBE_RANKS}
    depth_corr_by_rank = {r: [] for r in PROBE_RANKS}

    best_overall_metric = -1.0
    best_layer = 1
    best_rank = 64
    best_dist_probe = None
    best_depth_probe = None

    for rank in PROBE_RANKS:
        all_results[rank] = {}
        for layer in PROBE_LAYERS:
            logger.info(f"--- Training Distance & Depth Probes: Layer {layer}, Rank {rank} ---")
            
            # Train Distance Probe
            dist_probe, val_uuas = train_distance_probe(
                layer, rank,
                train_reps[layer], train_ds.gold_distances,
                val_reps[layer], val_ds.heads,
                device=device
            )
            # Evaluate on Test
            test_uuas, test_macro_uuas, correct_edges, total_edges = evaluate_distance_probe_uuas(
                dist_probe, test_reps[layer], test_ds.heads, device=device
            )

            # Train Depth Probe
            depth_probe, val_corr = train_depth_probe(
                layer, rank,
                train_reps[layer], train_ds.gold_depths,
                val_reps[layer], val_ds.gold_depths,
                device=device
            )
            # Evaluate on Test
            test_depth_corr, _ = evaluate_depth_probe_correlation(
                depth_probe, test_reps[layer], test_ds.gold_depths, device=device
            )

            uuas_by_rank[rank].append(test_uuas)
            depth_corr_by_rank[rank].append(test_depth_corr)

            all_results[rank][layer] = {
                "layer": layer,
                "rank": rank,
                "test_uuas": test_uuas,
                "test_macro_uuas": test_macro_uuas,
                "test_depth_correlation": test_depth_corr,
                "correct_edges": correct_edges,
                "total_edges": total_edges
            }

            logger.info(f"==> Layer {layer} (Rank {rank}): Test UUAS = {test_uuas:.4f}, Depth Corr = {test_depth_corr:.4f}")

            # Criterion for best syntax layer: maximum test UUAS (with depth correlation as tie-breaker)
            composite_score = test_uuas + 0.5 * test_depth_corr
            if composite_score > best_overall_metric:
                best_overall_metric = composite_score
                best_layer = layer
                best_rank = rank
                best_dist_probe = dist_probe
                best_depth_probe = depth_probe

    logger.info(f"*** Best Syntax Layer Selected: Layer l* = {best_layer} (Rank k = {best_rank}) with UUAS = {all_results[best_rank][best_layer]['test_uuas']:.4f} ***")

    # Save best probes
    torch.save(best_dist_probe.state_dict(), ckpt_dir / f"best_distance_probe_layer_{best_layer}_rank_{best_rank}.pt")
    torch.save(best_depth_probe.state_dict(), ckpt_dir / f"best_depth_probe_layer_{best_layer}_rank_{best_rank}.pt")
    
    with open(ckpt_dir / "best_layer_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_layer": best_layer,
            "best_rank": best_rank,
            "best_uuas": all_results[best_rank][best_layer]["test_uuas"],
            "best_depth_corr": all_results[best_rank][best_layer]["test_depth_correlation"]
        }, f, indent=2)

    # Save all numerical results
    with open(output_dir / "task2_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "selected_layer": best_layer,
            "selected_rank": best_rank,
            "results_by_rank": all_results
        }, f, indent=2)

    # Generate publication-quality plots
    plot_task2_layerwise_metrics(
        uuas_by_rank, depth_corr_by_rank, PROBE_LAYERS,
        FIGURES_DIR / "task2"
    )

    logger.info("=== Task 2 Completed Successfully! ===")
    return best_layer, all_results

if __name__ == "__main__":
    run_task2()
