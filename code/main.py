import argparse
import logging
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.data_utils import load_google_analogies, load_ud_gum_dataset
from common.data_utils import load_wordsim353_dataset
from config import (
    BERT_MODEL_NAME,
    CACHE_DIR,
    CHECKPOINTS_DIR,
    DATA_DIR,
    PROBE_EPOCHS,
    PROBE_LAYERS,
    PROBE_RANKS,
    RESULTS_DIR,
)
from scripts.download_data import download_all
from task1.main import run_task1
from task2.main import run_task2
from task3.main import run_task3

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project 5: static and contextual distributional semantics."
    )
    parser.add_argument(
        "--task", choices=("1", "2", "3", "all"), default="all", help="Task to run."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Data directory. Local mode expects ud_gum/train.conllu, validation.conllu, "
        "test.conllu, wordsim353.csv, and questions-words.txt.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--model-name", default=BERT_MODEL_NAME)
    parser.add_argument("--device", default=None, help="For example cpu or cuda.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--download", action="store_true", help="Download all resources first.")
    parser.add_argument("--epochs", type=int, default=PROBE_EPOCHS)
    parser.add_argument("--layers", type=int, nargs="+", default=list(PROBE_LAYERS))
    parser.add_argument("--ranks", type=int, nargs="+", default=list(PROBE_RANKS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.download:
        download_all(data_dir=args.data_dir, model_name=args.model_name)

    data_path = args.data_dir / "ud_gum" if (args.data_dir / "ud_gum").exists() else args.data_dir
    needs_splits = args.task in {"1", "2", "3", "all"}
    splits = (
        load_ud_gum_dataset(data_path, local_files_only=args.local_files_only)
        if needs_splits
        else None
    )
    needs_benchmarks = args.task in {"1", "3", "all"}
    wordsim = (
        load_wordsim353_dataset(
            args.data_dir / "wordsim353.csv", local_files_only=args.local_files_only
        )
        if needs_benchmarks
        else None
    )
    analogies = (
        load_google_analogies(
            args.data_dir / "questions-words.txt",
            local_files_only=args.local_files_only,
        )
        if needs_benchmarks
        else None
    )
    sample_indices_path = args.data_dir / "structural_sample_indices.json"

    if args.task in {"1", "all"}:
        run_task1(
            splits=splits,
            wordsim_pairs=wordsim,
            analogy_questions=analogies,
            output_dir=args.output_dir / "task1",
            checkpoints_dir=args.checkpoints_dir / "task1",
            sample_indices_path=sample_indices_path,
        )

    selected_layer: int | None = None
    if args.task in {"2", "all"}:
        selected_layer, _ = run_task2(
            splits=splits,
            layers=args.layers,
            ranks=args.ranks,
            epochs=args.epochs,
            output_dir=args.output_dir / "task2",
            checkpoints_dir=args.checkpoints_dir / "task2",
            figures_dir=args.output_dir / "figures",
            cache_dir=args.cache_dir,
            model_name=args.model_name,
            local_files_only=args.local_files_only,
            device=args.device,
        )

    if args.task == "3":
        task2_result = args.output_dir / "task2" / "task2_results.json"
        if task2_result.exists():
            import json

            selected_layer = int(
                json.loads(task2_result.read_text(encoding="utf-8"))["selected_layer"]
            )
    if args.task in {"3", "all"}:
        run_task3(
            splits=splits,
            wordsim_pairs=wordsim,
            analogy_questions=analogies,
            selected_layer=selected_layer,
            output_dir=args.output_dir / "task3",
            checkpoints_dir=args.checkpoints_dir / "task3",
            task1_results_path=args.output_dir / "task1" / "task1_results.json",
            task2_results_path=args.output_dir / "task2" / "task2_results.json",
            sample_indices_path=sample_indices_path,
            cache_dir=args.cache_dir,
            model_name=args.model_name,
            local_files_only=args.local_files_only,
            device=args.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
