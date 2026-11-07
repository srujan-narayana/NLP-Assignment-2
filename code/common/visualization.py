from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_task2_layerwise_metrics(
    results: dict[str, dict[str, dict[str, float]]],
    layers: list[int],
    save_path: str | Path,
) -> tuple[Path, Path]:
    prefix = Path(save_path)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    uuas_path = prefix.with_name(prefix.name + "_uuas_vs_layer.png")
    depth_path = prefix.with_name(prefix.name + "_depth_spearman_vs_layer.png")

    for field, title, ylabel, target in (
        ("test_uuas", "Task 2: Test UUAS by BERT layer", "UUAS", uuas_path),
        (
            "test_depth_spearman",
            "Task 2: Test depth correlation by BERT layer",
            "Spearman correlation",
            depth_path,
        ),
    ):
        fig, axis = plt.subplots(figsize=(8, 5))
        for rank, layer_results in results.items():
            values = [layer_results[str(layer)][field] for layer in layers]
            axis.plot(layers, values, marker="o", label=f"rank {rank}")
        axis.set(title=title, xlabel="BERT layer", ylabel=ylabel)
        axis.set_xticks(layers)
        axis.grid(True, linestyle="--", alpha=0.45)
        axis.legend()
        fig.tight_layout()
        fig.savefig(target, dpi=180)
        plt.close(fig)
    return uuas_path, depth_path
