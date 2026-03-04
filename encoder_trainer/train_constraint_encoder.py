from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

from trainer_config import trainer_config

from sentence_transformers import InputExample, SentenceTransformer, losses, models
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_examples(rows: list[dict]) -> list[InputExample]:
    examples: list[InputExample] = []
    for row in rows:
        q = row.get("query", "").strip()
        p = row.get("positive", "").strip()
        n = row.get("hard_negative", "").strip()
        if q and p and n:
            examples.append(InputExample(texts=[q, p, n]))
    return examples


def main() -> None:
    set_seed(int(trainer_config["seed"]))

    rows = read_jsonl(Path(str(trainer_config["train_file"])))
    examples = build_examples(rows)
    if not examples:
        raise RuntimeError(f"No valid training examples from {trainer_config['train_file']}")

    transformer = models.Transformer(
        str(trainer_config["base_model"]),
        max_seq_length=int(trainer_config["max_seq_len"]),
    )
    pooling = models.Pooling(transformer.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[transformer, pooling])

    loader = DataLoader(examples, shuffle=True, batch_size=int(trainer_config["batch_size"]))
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    warmup_steps = int(len(loader) * int(trainer_config["epochs"]) * 0.1)
    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=int(trainer_config["epochs"]),
        warmup_steps=warmup_steps,
        output_path=str(trainer_config["output_dir"]),
        show_progress_bar=True,
    )

    print(f"Training complete. Saved model to: {trainer_config['output_dir']}")


if __name__ == "__main__":
    main()
