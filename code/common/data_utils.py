import csv
import logging
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from config import (
    DATA_DIR,
    DATASET_CONFIG,
    DATASET_REPOSITORIES,
    GOOGLE_ANALOGY_URL,
    WORDSIM_DATASET,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UDSentence:
    tokens: tuple[str, ...]
    heads: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.heads):
            raise ValueError("A UD sentence must have one head value per token.")


def _sentence_from_row(row: dict[str, Any], split: str, index: int) -> UDSentence:
    if "tokens" not in row or "head" not in row:
        raise ValueError(
            f"{split} row {index} is missing required UD fields 'tokens' and 'head'."
        )
    tokens = tuple(str(token) for token in row["tokens"])
    try:
        heads = tuple(int(head) for head in row["head"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{split} row {index} has a non-integer dependency head.") from exc
    if len(tokens) != len(heads):
        raise ValueError(
            f"{split} row {index} has {len(tokens)} tokens and {len(heads)} heads."
        )
    return UDSentence(tokens=tokens, heads=heads)


def parse_conllu(path: str | Path) -> list[UDSentence]:
    """Read a CoNLL-U file while keeping only syntactic word rows."""
    sentences: list[UDSentence] = []
    tokens: list[str] = []
    heads: list[int] = []

    def flush() -> None:
        if tokens:
            sentences.append(UDSentence(tuple(tokens), tuple(heads)))
            tokens.clear()
            heads.clear()

    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                flush()
                continue
            if line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) < 7:
                raise ValueError(
                    f"{path}:{line_number}: expected at least 7 CoNLL-U columns."
                )
            token_id = columns[0]
            if not token_id.isdigit():
                # Multiword-token ranges (1-2) and empty nodes (1.1) are not words.
                continue
            try:
                head = int(columns[6])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid HEAD value {columns[6]!r}."
                ) from exc
            tokens.append(columns[1])
            heads.append(head)
    flush()
    return sentences


def _load_local_splits(data_dir: Path) -> dict[str, list[UDSentence]] | None:
    train_path = data_dir / "train.conllu"
    test_path = data_dir / "test.conllu"
    validation_path = data_dir / "validation.conllu"
    dev_path = data_dir / "dev.conllu"
    present = [path for path in (train_path, validation_path, dev_path, test_path) if path.exists()]
    if present and (not train_path.exists() or not test_path.exists()):
        raise FileNotFoundError(
            f"Local GUM data under {data_dir} is incomplete; train.conllu and "
            "test.conllu are required."
        )
    if not train_path.exists() or not test_path.exists():
        return None

    if not validation_path.exists():
        validation_path = dev_path
    if not validation_path.exists():
        raise FileNotFoundError(
            f"Expected validation.conllu or dev.conllu alongside {train_path}."
        )
    return {
        "train": parse_conllu(train_path),
        "validation": parse_conllu(validation_path),
        "test": parse_conllu(test_path),
    }


def load_ud_gum_dataset(
    data_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> dict[str, list[UDSentence]]:
    """Load all official GUM splits, either from local CoNLL-U files or Hugging Face."""
    root = Path(data_dir) if data_dir is not None else DATA_DIR / "ud_gum"
    local = _load_local_splits(root)
    if local is not None:
        return local
    if local_files_only:
        raise FileNotFoundError(
            "Local-only mode requires train.conllu, validation.conllu (or dev.conllu), "
            f"and test.conllu under {root}."
        )

    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Loading UD English GUM requires the pinned 'datasets' dependency."
        ) from exc

    errors: list[str] = []
    for repository in DATASET_REPOSITORIES:
        try:
            options: dict[str, Any] = {}
            if local_files_only:
                options["download_config"] = DownloadConfig(local_files_only=True)
            logger.info("Loading %s with configuration %s.", repository, DATASET_CONFIG)
            dataset = load_dataset(repository, DATASET_CONFIG, **options)
            raw_splits = dict(dataset)
            if "validation" not in raw_splits and "dev" in raw_splits:
                raw_splits["validation"] = raw_splits["dev"]
            required = ("train", "validation", "test")
            missing = [name for name in required if name not in raw_splits]
            if missing:
                raise ValueError(f"Dataset is missing required splits: {missing}")
            return {
                split: [
                    _sentence_from_row(row, split, index)
                    for index, row in enumerate(raw_splits[split])
                ]
                for split in required
            }
        except Exception as exc:
            errors.append(f"{repository}: {exc}")

    detail = "\n".join(errors)
    raise RuntimeError(
        "Could not load the UD English GUM dataset. Check network access or place "
        "train.conllu, validation.conllu (or dev.conllu), and test.conllu in "
        f"{root}.\n{detail}"
    )


def load_wordsim353_dataset(
    path: str | Path | None = None,
    local_files_only: bool = False,
) -> list[tuple[str, str, float]]:
    """Load the shared 353-pair benchmark from CSV or its Hugging Face dataset."""
    csv_path = Path(path) if path is not None else DATA_DIR / "wordsim353.csv"
    if csv_path.exists():
        pairs: list[tuple[str, str, float]] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"{csv_path} has no CSV header.")
            fields = {name.casefold().strip(): name for name in reader.fieldnames}
            word1_key = next(
                (fields[key] for key in ("word1", "word 1", "word_1") if key in fields),
                None,
            )
            word2_key = next(
                (fields[key] for key in ("word2", "word 2", "word_2") if key in fields),
                None,
            )
            score_key = next(
                (
                    fields[key]
                    for key in ("similarity", "human (mean)", "score")
                    if key in fields
                ),
                None,
            )
            if not word1_key or not word2_key or not score_key:
                raise ValueError(
                    f"{csv_path} needs word1, word2, and similarity/score columns."
                )
            for row in reader:
                pairs.append(
                    (
                        str(row[word1_key]).strip().lower(),
                        str(row[word2_key]).strip().lower(),
                        float(row[score_key]),
                    )
                )
        return pairs

    if local_files_only:
        raise FileNotFoundError(
            f"WordSim-353 is not present at {csv_path} and local_files_only is enabled."
        )

    try:
        from datasets import load_dataset

        dataset = load_dataset(WORDSIM_DATASET)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load WordSim-353. Put wordsim353.csv in {csv_path.parent} "
            f"or allow Hugging Face access. Original error: {exc}"
        ) from exc

    split_name = "train" if "train" in dataset else next(iter(dataset))
    pairs = []
    for row in dataset[split_name]:
        normalized = {str(key).casefold().strip(): value for key, value in row.items()}
        w1 = next(
            (normalized[key] for key in ("word1", "word 1", "word_1") if key in normalized),
            None,
        )
        w2 = next(
            (normalized[key] for key in ("word2", "word 2", "word_2") if key in normalized),
            None,
        )
        score = next(
            (
                normalized[key]
                for key in ("similarity", "human (mean)", "score")
                if key in normalized
            ),
            None,
        )
        if w1 is None or w2 is None or score is None:
            raise ValueError("WordSim-353 dataset has an unrecognized schema.")
        pairs.append((str(w1).strip().lower(), str(w2).strip().lower(), float(score)))
    return pairs


def ensure_google_analogy_dataset(
    path: str | Path | None = None, local_files_only: bool = False
) -> Path:
    """Return the Google analogy file, downloading it atomically if it is absent."""
    target = Path(path) if path is not None else DATA_DIR / "questions-words.txt"
    if target.exists():
        return target
    if local_files_only:
        raise FileNotFoundError(
            f"Google analogy data is not present at {target} and local_files_only is enabled."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="questions-words-", suffix=".tmp", dir=target.parent, delete=False
        ) as temp:
            temp_name = temp.name
        urllib.request.urlretrieve(GOOGLE_ANALOGY_URL, temp_name)
        if Path(temp_name).stat().st_size == 0:
            raise OSError("Downloaded analogy file is empty.")
        os.replace(temp_name, target)
    except Exception as exc:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()
        raise RuntimeError(
            f"Could not download Google analogies to {target}. "
            f"Place questions-words.txt there manually. Original error: {exc}"
        ) from exc
    return target


def load_google_analogies(
    path: str | Path | None = None,
    local_files_only: bool = False,
) -> list[tuple[str, str, str, str, str]]:
    """Read Google analogy questions as (category, a, b, c, expected) tuples."""
    analogy_path = ensure_google_analogy_dataset(path, local_files_only=local_files_only)
    questions: list[tuple[str, str, str, str, str]] = []
    category = "uncategorized"
    with analogy_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(":"):
                category = line[1:].strip() or "uncategorized"
                continue
            words = line.lower().split()
            if len(words) != 4:
                logger.warning("Skipping malformed analogy line %d.", line_number)
                continue
            questions.append((category, words[0], words[1], words[2], words[3]))
    return questions


def ensure_output_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
