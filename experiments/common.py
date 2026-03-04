from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
REPORTS_DIR = OUTPUTS_DIR / "reports"


def ensure_project_dirs() -> None:
    for path in (
        RAW_DIR,
        PROCESSED_DIR,
        FIGURES_DIR,
        CHECKPOINTS_DIR,
        REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


@dataclass
class PairwiseAccuracy:
    total: int
    correct: int
    avg_margin: float
    median_margin: float

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "avg_margin": self.avg_margin,
            "median_margin": self.median_margin,
        }


def pairwise_accuracy(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> PairwiseAccuracy:
    if len(pos_scores) != len(neg_scores):
        raise ValueError("pos_scores and neg_scores must have the same length.")
    margins = [p - n for p, n in zip(pos_scores, neg_scores)]
    correct = sum(1 for m in margins if m > 0)
    avg = float(np.mean(margins)) if margins else 0.0
    med = float(np.median(margins)) if margins else 0.0
    return PairwiseAccuracy(total=len(margins), correct=correct, avg_margin=avg, median_margin=med)


def safe_float(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return float(x)


def load_sentence_encoder(model_name_or_path: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name_or_path)
    tokenizer = None
    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        tokenizer = model.tokenizer
    elif len(model) > 0 and hasattr(model[0], "tokenizer"):
        tokenizer = model[0].tokenizer

    # Local causal LMs often miss pad token; SentenceTransformer.encode expects padded batches.
    if tokenizer is not None and getattr(tokenizer, "pad_token", None) is None:
        eos = getattr(tokenizer, "eos_token", None)
        if eos is not None:
            tokenizer.pad_token = eos
            if hasattr(model, "tokenizer") and model.tokenizer is not None:
                model.tokenizer.pad_token = eos
            if len(model) > 0 and hasattr(model[0], "auto_model"):
                cfg = model[0].auto_model.config
                if getattr(cfg, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
                    cfg.pad_token_id = tokenizer.pad_token_id
    return model
