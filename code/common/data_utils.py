import os
import urllib.request
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

from config import DATA_DIR, GOOGLE_ANALOGY_PATH, GOOGLE_ANALOGY_URL, DATASET_NAME, DATASET_CONFIG, WORDSIM353_DATASET

logger = logging.getLogger(__name__)

def load_ud_gum_dataset() -> Dict[str, Any]:
    """Load the UD English GUM dataset splits."""
    from datasets import load_dataset
    logger.info(f"Loading {DATASET_NAME} with config {DATASET_CONFIG}...")
    try:
        ds = load_dataset(DATASET_NAME, DATASET_CONFIG)
    except Exception as e:
        logger.warning(f"Failed with {DATASET_NAME}, trying fallback universal_dependencies: {e}")
        ds = load_dataset("universal_dependencies", DATASET_CONFIG)

    # Normalize split names: ensure both 'dev' and 'validation' are accessible
    result = {}
    for split_key in ds.keys():
        result[split_key] = ds[split_key]
    if "validation" not in result and "dev" in result:
        result["validation"] = result["dev"]
    if "dev" not in result and "validation" in result:
        result["dev"] = result["validation"]

    logger.info(f"Loaded splits: train={len(result['train'])}, validation={len(result['validation'])}, test={len(result['test'])}")
    return result

def ensure_google_analogy_dataset() -> Path:
    """Download the Google Analogy dataset (questions-words.txt) if not present."""
    if not GOOGLE_ANALOGY_PATH.exists():
        logger.info(f"Downloading Google Analogy questions from {GOOGLE_ANALOGY_URL}...")
        urllib.request.urlretrieve(GOOGLE_ANALOGY_URL, str(GOOGLE_ANALOGY_PATH))
        logger.info(f"Downloaded questions-words.txt to {GOOGLE_ANALOGY_PATH}")
    return GOOGLE_ANALOGY_PATH

def load_google_analogies() -> List[Tuple[str, str, str, str, str]]:
    """
    Parse Google analogy questions-words.txt.
    Returns:
        List of (category, word_a, word_b, word_c, word_d)
    """
    path = ensure_google_analogy_dataset()
    questions = []
    current_category = "default"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                current_category = line[1:].strip()
            else:
                parts = line.lower().split()
                if len(parts) == 4:
                    questions.append((current_category, parts[0], parts[1], parts[2], parts[3]))
    return questions

def load_wordsim353_dataset() -> List[Tuple[str, str, float]]:
    """
    Load WordSim-353 benchmark.
    Returns:
        List of (word1, word2, score)
    """
    from datasets import load_dataset
    try:
        ds = load_dataset(WORDSIM353_DATASET)
        split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
        pairs = []
        for row in split:
            # Handle possible key naming conventions
            w1 = str(row.get("word1") or row.get("Word 1") or row.get("Word1")).lower().strip()
            w2 = str(row.get("word2") or row.get("Word 2") or row.get("Word2")).lower().strip()
            score_val = row.get("similarity") or row.get("Human (Mean)") or row.get("score")
            score = float(score_val)
            pairs.append((w1, w2, score))
        logger.info(f"Loaded {len(pairs)} WordSim-353 pairs via Hugging Face.")
        return pairs
    except Exception as e:
        logger.warning(f"Error loading WordSim-353 via Hugging Face ({e}), trying direct CSV download fallback...")
        csv_path = DATA_DIR / "wordsim353.csv"
        if not csv_path.exists():
            url = "https://raw.githubusercontent.com/enus5three/WordSim353/master/combined.csv"
            urllib.request.urlretrieve(url, str(csv_path))
        pairs = []
        with open(csv_path, "r", encoding="utf-8") as f:
            import csv
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    pairs.append((row[0].lower().strip(), row[1].lower().strip(), float(row[2])))
        return pairs
