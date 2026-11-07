import csv
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.data_utils import ensure_google_analogy_dataset, load_wordsim353_dataset
from config import BERT_MODEL_NAME, DATA_DIR, DATASET_CONFIG, DATASET_REPOSITORIES


def _write_conllu(path: Path, rows: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            tokens = row["tokens"]
            heads = row["head"]
            if len(tokens) != len(heads):
                raise ValueError("A dataset row contains a mismatched token/head count.")
            for index, (token, head) in enumerate(zip(tokens, heads), start=1):
                stream.write(
                    f"{index}\t{token}\t_\t_\t_\t_\t{int(head)}\t_\t_\t_\n"
                )
            stream.write("\n")


def download_all(
    data_dir: str | Path = DATA_DIR,
    model_name: str = BERT_MODEL_NAME,
) -> None:
    root = Path(data_dir)
    try:
        from datasets import load_dataset
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before downloading resources.") from exc

    errors: list[str] = []
    dataset = None
    for repository in DATASET_REPOSITORIES:
        try:
            dataset = load_dataset(repository, DATASET_CONFIG)
            break
        except Exception as exc:
            errors.append(f"{repository}: {exc}")
    if dataset is None:
        raise RuntimeError("Could not download UD English GUM:\n" + "\n".join(errors))

    raw_splits = dict(dataset)
    if "validation" not in raw_splits and "dev" in raw_splits:
        raw_splits["validation"] = raw_splits["dev"]
    for split in ("train", "validation", "test"):
        if split not in raw_splits:
            raise ValueError(f"Downloaded GUM data is missing the {split} split.")
        filename = "validation.conllu" if split == "validation" else f"{split}.conllu"
        _write_conllu(root / "ud_gum" / filename, raw_splits[split])

    wordsim = load_wordsim353_dataset(root / "wordsim353.csv")
    with (root / "wordsim353.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["word1", "word2", "similarity"])
        writer.writerows(wordsim)
    ensure_google_analogy_dataset(root / "questions-words.txt")
    AutoTokenizer.from_pretrained(model_name)
    AutoModel.from_pretrained(model_name)
    print(f"GUM splits and benchmarks saved under {root}.")
    print(f"BERT checkpoint {model_name} is cached and ready.")


if __name__ == "__main__":
    download_all()
