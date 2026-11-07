import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from common.data_utils import load_ud_gum_dataset
from task2.gold_structures import build_gold_structures

logger = logging.getLogger(__name__)

class ProbingDataset:
    """
    Manages sentences, tokens, heads, gold distances, and gold depths.
    Uses the assignment specification splits:
    - Train: ~2,300 sentences
    - Validation: ~780 sentences
    - Test: ~780 sentences
    """
    def __init__(self, split_name: str):
        self.split_name = split_name
        ds = load_ud_gum_dataset()
        raw_split = ds[split_name]

        # Use assignment specified split limits
        if split_name == "train":
            raw_split = raw_split.select(range(min(2300, len(raw_split))))
        elif split_name in ["validation", "dev"]:
            raw_split = raw_split.select(range(min(780, len(raw_split))))
        elif split_name == "test":
            raw_split = raw_split.select(range(min(780, len(raw_split))))

        self.sentences_tokens = []
        self.heads = []
        self.gold_distances = []
        self.gold_depths = []

        logger.info(f"Processing gold dependency structures for split '{split_name}' ({len(raw_split)} sentences)...")
        for item in raw_split:
            tokens = [str(t).lower() for t in item["tokens"]]
            h = item["head"]
            dist_mat, depth_arr, _ = build_gold_structures(h)

            self.sentences_tokens.append(tokens)
            self.heads.append(h)
            self.gold_distances.append(torch.tensor(dist_mat, dtype=torch.float32))
            self.gold_depths.append(torch.tensor(depth_arr, dtype=torch.float32))

        logger.info(f"Split '{split_name}' ready with {len(self.sentences_tokens)} processed sentences.")

    def __len__(self):
        return len(self.sentences_tokens)
