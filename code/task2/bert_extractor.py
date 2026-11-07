import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from common.data_utils import UDSentence
from config import BERT_MODEL_NAME, CACHE_DIR, MAX_BERT_TOKENS

logger = logging.getLogger(__name__)
CACHE_SCHEMA = 1


@dataclass
class FeatureStore:
    directory: Path
    lengths: list[int]
    hidden_size: int

    @property
    def total_words(self) -> int:
        return sum(self.lengths)

    def layer(self, layer_number: int) -> list[torch.Tensor]:
        path = self.directory / f"layer_{layer_number}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Cached BERT layer {layer_number} is missing: {path}")
        values = np.load(path, mmap_mode="r")
        result: list[torch.Tensor] = []
        offset = 0
        for length in self.lengths:
            # Copy the small sentence slice so PyTorch never wraps a read-only memmap.
            result.append(torch.from_numpy(np.array(values[offset : offset + length], copy=True)))
            offset += length
        if offset != values.shape[0]:
            raise ValueError(f"Feature cache has an invalid word count: {path}")
        return result


def sentence_fingerprint(
    sentences: Sequence[UDSentence],
    model_name: str,
    model_revision: str,
    max_tokens: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"{CACHE_SCHEMA}|{model_name}|{model_revision}|max_tokens={max_tokens}".encode(
            "utf-8"
        )
    )
    for sentence in sentences:
        digest.update(json.dumps(sentence.tokens, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class BertLayerExtractor:
    def __init__(
        self,
        model_name: str = BERT_MODEL_NAME,
        device: str | torch.device | None = None,
        local_files_only: bool = False,
        tokenizer: Any = None,
        model: Any = None,
        max_tokens: int = MAX_BERT_TOKENS,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_name = model_name
        self.max_tokens = int(max_tokens)
        if self.max_tokens < 3:
            raise ValueError("max_tokens must allow CLS, at least one piece, and SEP.")
        if (tokenizer is None) != (model is None):
            raise ValueError("Inject both tokenizer and model, or neither.")
        if tokenizer is None:
            try:
                from transformers import AutoModel, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, use_fast=True, local_files_only=local_files_only
                )
                model = AutoModel.from_pretrained(
                    model_name, local_files_only=local_files_only
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load {model_name}. Download it first or allow model access. "
                    f"Original error: {exc}"
                ) from exc
        self.tokenizer = tokenizer
        if self.tokenizer.cls_token_id is None or self.tokenizer.sep_token_id is None:
            raise ValueError("The selected tokenizer must define CLS and SEP token IDs.")
        self.model = model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.hidden_size = int(self.model.config.hidden_size)
        self.model_revision = str(
            getattr(self.model.config, "_commit_hash", None) or "unspecified-revision"
        )

    def _word_segments(self, words: Sequence[str]) -> list[list[tuple[int, list[int]]]]:
        """Chunk at word boundaries and preserve all pieces, including very long tokens."""
        capacity = self.max_tokens - 2
        chunks: list[list[tuple[int, list[int]]]] = []
        current: list[tuple[int, list[int]]] = []
        used = 0
        for word_index, word in enumerate(words):
            pieces = self.tokenizer.encode(str(word), add_special_tokens=False)
            if not pieces:
                unknown = self.tokenizer.unk_token_id
                if unknown is None:
                    raise ValueError(f"Tokenizer produced no pieces for {word!r}.")
                pieces = [int(unknown)]
            remaining = [int(piece) for piece in pieces]
            while remaining:
                available = capacity - used
                if available == 0:
                    chunks.append(current)
                    current, used = [], 0
                    available = capacity
                take = min(available, len(remaining))
                segment = remaining[:take]
                remaining = remaining[take:]
                current.append((word_index, segment))
                used += len(segment)
                if remaining:
                    chunks.append(current)
                    current, used = [], 0
        if current:
            chunks.append(current)
        return chunks

    @torch.no_grad()
    def _extract_sentence(
        self, words: Sequence[str], requested_layers: Sequence[int]
    ) -> dict[int, np.ndarray]:
        layer_sums = {
            layer: np.zeros((len(words), self.hidden_size), dtype=np.float64)
            for layer in requested_layers
        }
        piece_counts = np.zeros(len(words), dtype=np.int64)
        chunks = self._word_segments(words)
        for chunk in chunks:
            input_ids = [int(self.tokenizer.cls_token_id)]
            spans: list[tuple[int, int, int]] = []
            for word_index, pieces in chunk:
                start = len(input_ids)
                input_ids.extend(pieces)
                spans.append((word_index, start, len(input_ids)))
            input_ids.append(int(self.tokenizer.sep_token_id))
            tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            output = self.model(input_ids=tensor, output_hidden_states=True)
            for word_index, start, end in spans:
                piece_counts[word_index] += end - start
                for layer in requested_layers:
                    piece_hidden = output.hidden_states[layer][0, start:end]
                    layer_sums[layer][word_index] += (
                        piece_hidden.float().sum(dim=0).cpu().numpy()
                    )

        result: dict[int, np.ndarray] = {}
        for layer in requested_layers:
            result[layer] = np.divide(
                layer_sums[layer],
                piece_counts[:, None],
                out=np.zeros_like(layer_sums[layer]),
                where=piece_counts[:, None] > 0,
            ).astype(np.float32)
        return result

    def ensure_features(
        self,
        split_name: str,
        sentences: Sequence[UDSentence],
        layers: Sequence[int] = tuple(range(1, 13)),
        cache_dir: str | Path = CACHE_DIR,
    ) -> FeatureStore:
        requested = sorted(set(int(layer) for layer in layers))
        maximum_layer = int(getattr(self.model.config, "num_hidden_layers", 12))
        if not requested or any(layer < 1 or layer > maximum_layer for layer in requested):
            raise ValueError(f"Requested BERT layers must be in 1..{maximum_layer}.")
        lengths = [len(sentence.tokens) for sentence in sentences]
        total_words = sum(lengths)
        if total_words <= 0:
            raise ValueError(f"The {split_name} split contains no words.")
        signature = sentence_fingerprint(
            sentences, self.model_name, self.model_revision, self.max_tokens
        )
        directory = Path(cache_dir) / f"{split_name}_{signature[:20]}"
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
        metadata_matches = (
            manifest.get("schema") == CACHE_SCHEMA
            and manifest.get("fingerprint") == signature
            and manifest.get("lengths") == lengths
            and manifest.get("hidden_size") == self.hidden_size
        )
        cached_layers: set[int] = set()
        if metadata_matches:
            cached_layers = set(manifest.get("layers", []))
            cached_layers = {
                layer
                for layer in cached_layers
                if (directory / f"layer_{layer}.npy").exists()
            }
        missing = [layer for layer in requested if layer not in cached_layers]
        if missing:
            maps: dict[int, np.memmap] = {}
            try:
                for layer in missing:
                    maps[layer] = np.lib.format.open_memmap(
                        directory / f"layer_{layer}.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=(total_words, self.hidden_size),
                    )
                offset = 0
                for sentence_index, sentence in enumerate(sentences):
                    extracted = self._extract_sentence(sentence.tokens, missing)
                    end = offset + len(sentence.tokens)
                    for layer, values in extracted.items():
                        maps[layer][offset:end] = values
                    offset = end
                    if (sentence_index + 1) % 250 == 0:
                        logger.info(
                            "Extracted %d/%d sentences for %s.",
                            sentence_index + 1,
                            len(sentences),
                            split_name,
                        )
                for value in maps.values():
                    value.flush()
                del maps
            except Exception:
                for layer in missing:
                    path = directory / f"layer_{layer}.npy"
                    if path.exists():
                        path.unlink()
                raise
            cached_layers.update(missing)
            new_manifest = {
                "schema": CACHE_SCHEMA,
                "fingerprint": signature,
                "split": split_name,
                "model_name": self.model_name,
                "model_revision": self.model_revision,
                "hidden_size": self.hidden_size,
                "dtype": "float32",
                "lengths": lengths,
                "layers": sorted(cached_layers),
            }
            handle, temp_path = tempfile.mkstemp(
                prefix="manifest-", suffix=".tmp", dir=directory
            )
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as stream:
                    json.dump(new_manifest, stream, indent=2)
                os.replace(temp_path, manifest_path)
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

        return FeatureStore(directory, lengths, self.hidden_size)
