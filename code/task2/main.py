import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from common.data_utils import UDSentence, load_ud_gum_dataset
from common.reproducibility import set_seed
from common.visualization import plot_task2_layerwise_metrics
from config import (
    BERT_MODEL_NAME,
    CACHE_DIR,
    CHECKPOINTS_DIR,
    PROBE_BATCH_SIZE,
    PROBE_EPOCHS,
    PROBE_LAYERS,
    PROBE_LR,
    PROBE_RANKS,
    RANDOM_SEED,
    RESULTS_DIR,
)
from task2.bert_extractor import BertLayerExtractor, FeatureStore
from task2.depth_probe import StructuralDepthProbe
from task2.distance_probe import StructuralDistanceProbe
from task2.evaluation import evaluate_depth_probe_correlation
from task2.evaluation import evaluate_distance_probe_uuas
from task2.gold_structures import build_gold_structures

logger = logging.getLogger(__name__)


def _collate(
    representations: Sequence[torch.Tensor],
    targets: Sequence[np.ndarray | torch.Tensor],
    indices: Sequence[int],
    task: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_reps = [representations[index] for index in indices]
    lengths = torch.tensor([rep.shape[0] for rep in batch_reps], device=device)
    max_length = int(lengths.max().item()) if len(batch_reps) else 0
    hidden_size = representations[0].shape[1]
    features = torch.zeros(
        (len(batch_reps), max_length, hidden_size), dtype=torch.float32, device=device
    )
    if task == "distance":
        gold = torch.zeros(
            (len(batch_reps), max_length, max_length), dtype=torch.float32, device=device
        )
    elif task == "depth":
        gold = torch.zeros((len(batch_reps), max_length), dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unknown probe task: {task}")

    for batch_index, (rep, target) in enumerate(zip(batch_reps, (targets[i] for i in indices))):
        length = rep.shape[0]
        features[batch_index, :length] = rep.to(device=device, dtype=torch.float32)
        target_tensor = (
            target.to(device=device, dtype=torch.float32)
            if isinstance(target, torch.Tensor)
            else torch.as_tensor(target, dtype=torch.float32, device=device)
        )
        if task == "distance":
            gold[batch_index, :length, :length] = target_tensor
        else:
            gold[batch_index, :length] = target_tensor
    return features, gold, lengths


def _distance_batch_loss(
    probe: StructuralDistanceProbe,
    features: torch.Tensor,
    gold: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    predicted = probe(features)
    positions = torch.arange(features.shape[1], device=features.device)
    valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
    pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
    absolute = (predicted - gold).abs() * pair_mask
    per_sentence = absolute.sum(dim=(1, 2)) / lengths.to(torch.float32).square()
    return per_sentence.mean()


def _depth_batch_loss(
    probe: StructuralDepthProbe,
    features: torch.Tensor,
    gold: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    predicted = probe(features)
    positions = torch.arange(features.shape[1], device=features.device)
    mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
    absolute = (predicted - gold).abs() * mask
    per_sentence = absolute.sum(dim=1) / lengths.to(torch.float32)
    return per_sentence.mean()


def _fit_probe(
    probe: StructuralDistanceProbe | StructuralDepthProbe,
    task: str,
    train_representations: Sequence[torch.Tensor],
    train_targets: Sequence[np.ndarray | torch.Tensor],
    validation_representations: Sequence[torch.Tensor],
    validation_targets: Sequence[np.ndarray | torch.Tensor],
    validation_sentences: Sequence[UDSentence],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    validation_interval: int = 5,
) -> tuple[StructuralDistanceProbe | StructuralDepthProbe, float]:
    if not train_representations:
        raise ValueError("Cannot train a probe on an empty training split.")
    if len(train_representations) != len(train_targets):
        raise ValueError("Training representations and gold targets must align.")
    if len(validation_representations) != len(validation_targets):
        raise ValueError("Validation representations and gold targets must align.")
    if task == "distance" and len(validation_representations) != len(validation_sentences):
        raise ValueError("Validation representations and sentences must align.")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Epochs, batch size, and learning rate must be positive.")

    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    interval = max(1, validation_interval)
    loss_function = _distance_batch_loss if task == "distance" else _depth_batch_loss

    for epoch in range(1, epochs + 1):
        probe.train()
        permutation = torch.randperm(len(train_representations)).tolist()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            features, gold, lengths = _collate(
                train_representations, train_targets, indices, task, device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(probe, features, gold, lengths)
            loss.backward()
            optimizer.step()

        should_validate = epoch % interval == 0 or epoch == epochs
        if not should_validate:
            continue
        if task == "distance":
            score = evaluate_distance_probe_uuas(
                probe, validation_representations, validation_sentences, device
            )["uuas"]
        else:
            score = evaluate_depth_probe_correlation(
                probe, validation_representations, validation_targets, device
            )["spearman_correlation"]
        if float(score) > best_score:
            best_score = float(score)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in probe.state_dict().items()
            }
        logger.info(
            "%s probe epoch %d/%d: validation score %.4f",
            task,
            epoch,
            epochs,
            score,
        )

    if best_state is not None:
        probe.load_state_dict(best_state)
    return probe, best_score


def train_distance_probe(
    train_representations: Sequence[torch.Tensor],
    train_distances: Sequence[np.ndarray | torch.Tensor],
    validation_representations: Sequence[torch.Tensor],
    validation_sentences: Sequence[UDSentence],
    rank: int,
    device: str | torch.device = "cpu",
    epochs: int = PROBE_EPOCHS,
    batch_size: int = PROBE_BATCH_SIZE,
    learning_rate: float = PROBE_LR,
) -> tuple[StructuralDistanceProbe, float]:
    probe = StructuralDistanceProbe(
        input_dim=train_representations[0].shape[1], probe_rank=rank
    ).to(device)
    targets = [build_gold_structures(sentence.heads)[0] for sentence in validation_sentences]
    trained, score = _fit_probe(
        probe,
        "distance",
        train_representations,
        train_distances,
        validation_representations,
        targets,
        validation_sentences,
        torch.device(device),
        epochs,
        learning_rate,
        batch_size,
    )
    return trained, score  # type: ignore[return-value]


def train_depth_probe(
    train_representations: Sequence[torch.Tensor],
    train_depths: Sequence[np.ndarray | torch.Tensor],
    validation_representations: Sequence[torch.Tensor],
    validation_depths: Sequence[np.ndarray | torch.Tensor],
    rank: int,
    device: str | torch.device = "cpu",
    epochs: int = PROBE_EPOCHS,
    batch_size: int = PROBE_BATCH_SIZE,
    learning_rate: float = PROBE_LR,
) -> tuple[StructuralDepthProbe, float]:
    probe = StructuralDepthProbe(
        input_dim=train_representations[0].shape[1], probe_rank=rank
    ).to(device)
    trained, score = _fit_probe(
        probe,
        "depth",
        train_representations,
        train_depths,
        validation_representations,
        validation_depths,
        [],
        torch.device(device),
        epochs,
        learning_rate,
        batch_size,
    )
    return trained, score  # type: ignore[return-value]


def _get_gold(splits: dict[str, list[UDSentence]]) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    return {
        split: [build_gold_structures(sentence.heads) for sentence in splits[split]]
        for split in ("train", "validation", "test")
    }


def run_task2(
    splits: dict[str, list[UDSentence]] | None = None,
    extractor: BertLayerExtractor | None = None,
    layers: Sequence[int] = PROBE_LAYERS,
    ranks: Sequence[int] = PROBE_RANKS,
    epochs: int = PROBE_EPOCHS,
    batch_size: int = PROBE_BATCH_SIZE,
    output_dir: str | Path | None = None,
    checkpoints_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    cache_dir: str | Path = CACHE_DIR,
    model_name: str = BERT_MODEL_NAME,
    local_files_only: bool = False,
    device: str | torch.device | None = None,
) -> tuple[int, dict[str, object]]:
    set_seed(RANDOM_SEED)
    splits = splits or load_ud_gum_dataset(local_files_only=local_files_only)
    required = ("train", "validation", "test")
    if any(not splits.get(split) for split in required):
        raise ValueError("Task 2 requires non-empty train, validation, and test splits.")
    if any(not sentence.tokens for split in required for sentence in splits[split]):
        raise ValueError("Task 2 does not accept empty sentences.")
    layers = tuple(int(layer) for layer in layers)
    ranks = tuple(int(rank) for rank in ranks)
    if not layers or not ranks:
        raise ValueError("At least one BERT layer and probe rank must be selected.")

    gold = _get_gold(splits)
    extractor = extractor or BertLayerExtractor(
        model_name=model_name, device=device, local_files_only=local_files_only
    )
    feature_stores: dict[str, FeatureStore] = {
        split: extractor.ensure_features(split, splits[split], layers, cache_dir)
        for split in required
    }
    execution_device = torch.device(device) if device is not None else extractor.device
    result_path = Path(output_dir) if output_dir is not None else RESULTS_DIR / "task2"
    checkpoint_path = (
        Path(checkpoints_dir) if checkpoints_dir is not None else CHECKPOINTS_DIR / "task2"
    )
    figure_path = Path(figures_dir) if figures_dir is not None else RESULTS_DIR / "figures"
    result_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    figure_path.mkdir(parents=True, exist_ok=True)

    result_by_rank: dict[str, dict[str, dict[str, float | int]]] = {}
    best_selection = (-float("inf"), -float("inf"))
    best_layer = layers[0]
    best_rank = ranks[0]
    best_distance_state: dict[str, torch.Tensor] | None = None
    best_depth_state: dict[str, torch.Tensor] | None = None

    for rank in ranks:
        result_by_rank[str(rank)] = {}
        for layer in layers:
            train_reps = feature_stores["train"].layer(layer)
            validation_reps = feature_stores["validation"].layer(layer)
            test_reps = feature_stores["test"].layer(layer)
            train_distances = [item[0] for item in gold["train"]]
            train_depths = [item[1] for item in gold["train"]]
            validation_distances = [item[0] for item in gold["validation"]]
            validation_depths = [item[1] for item in gold["validation"]]

            distance_probe = StructuralDistanceProbe(
                input_dim=feature_stores["train"].hidden_size, probe_rank=rank
            ).to(execution_device)
            distance_probe, validation_uuas = _fit_probe(
                distance_probe,
                "distance",
                train_reps,
                train_distances,
                validation_reps,
                validation_distances,
                splits["validation"],
                execution_device,
                epochs,
                PROBE_LR,
                batch_size,
            )
            depth_probe = StructuralDepthProbe(
                input_dim=feature_stores["train"].hidden_size, probe_rank=rank
            ).to(execution_device)
            depth_probe, _ = _fit_probe(
                depth_probe,
                "depth",
                train_reps,
                train_depths,
                validation_reps,
                validation_depths,
                splits["validation"],
                execution_device,
                epochs,
                PROBE_LR,
                batch_size,
            )

            validation_depth = evaluate_depth_probe_correlation(
                depth_probe, validation_reps, validation_depths, execution_device
            )
            test_distance = evaluate_distance_probe_uuas(
                distance_probe, test_reps, splits["test"], execution_device
            )
            test_depth = evaluate_depth_probe_correlation(
                depth_probe, test_reps, [item[1] for item in gold["test"]], execution_device
            )
            result_by_rank[str(rank)][str(layer)] = {
                "layer": layer,
                "rank": rank,
                "validation_uuas": float(validation_uuas),
                "validation_depth_spearman": validation_depth["spearman_correlation"],
                "test_uuas": test_distance["uuas"],
                "test_macro_uuas": test_distance["macro_uuas"],
                "test_depth_spearman": test_depth["spearman_correlation"],
                "test_depth_macro_spearman": test_depth["macro_spearman"],
                "correct_edges": test_distance["correct_edges"],
                "gold_edges": test_distance["gold_edges"],
            }
            selection = (
                float(validation_uuas),
                float(validation_depth["spearman_correlation"]),
            )
            if selection > best_selection:
                best_selection = selection
                best_layer, best_rank = layer, rank
                best_distance_state = {
                    key: value.detach().cpu().clone()
                    for key, value in distance_probe.state_dict().items()
                }
                best_depth_state = {
                    key: value.detach().cpu().clone()
                    for key, value in depth_probe.state_dict().items()
                }
            logger.info(
                "Layer %d rank %d: validation UUAS %.4f, test UUAS %.4f, "
                "test depth Spearman %.4f.",
                layer,
                rank,
                validation_uuas,
                test_distance["uuas"],
                test_depth["spearman_correlation"],
            )
            del train_reps, validation_reps, test_reps
            del distance_probe, depth_probe

    if best_distance_state is None or best_depth_state is None:
        raise RuntimeError("No probe configuration completed.")
    torch.save(
        best_distance_state,
        checkpoint_path / f"best_distance_probe_layer_{best_layer}_rank_{best_rank}.pt",
    )
    torch.save(
        best_depth_state,
        checkpoint_path / f"best_depth_probe_layer_{best_layer}_rank_{best_rank}.pt",
    )
    layer_info = {
        "selected_layer": best_layer,
        "selected_rank": best_rank,
        "selection_metric": "validation UUAS, with validation depth Spearman as tie-breaker",
        "validation_uuas": best_selection[0],
        "validation_depth_spearman": best_selection[1],
    }
    (checkpoint_path / "best_layer_info.json").write_text(
        json.dumps(layer_info, indent=2), encoding="utf-8"
    )

    summary: dict[str, object] = {
        **layer_info,
        "model_name": model_name,
        "layers": list(layers),
        "ranks": list(ranks),
        "epochs": epochs,
        "results_by_rank": result_by_rank,
    }
    (result_path / "task2_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_task2_layerwise_metrics(
        result_by_rank,
        list(layers),
        figure_path / "task2",
    )
    return best_layer, summary
