from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses, models
from torch.utils.data import DataLoader

from common import ensure_project_dirs, read_jsonl, set_seed


def load_trainer_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "encoder_trainer" / "trainer_config.py"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("coder_trainer_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config module from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = getattr(module, "trainer_config", None)
    if not isinstance(cfg, dict):
        raise TypeError("`trainer_config` must be a dict in CoDeR/encoder_trainer/trainer_config.py")
    return cfg


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
    cfg = load_trainer_config()
    ensure_project_dirs()
    set_seed(int(cfg.get("seed", 42)))

    cuda_visible_devices = str(cfg.get("cuda_visible_devices", ""))
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    train_file = str(cfg.get("train_file", "data/processed/train_triplets.jsonl"))
    base_model = str(cfg.get("base_model", "sentence-transformers/all-MiniLM-L6-v2"))
    output_dir = str(cfg.get("output_dir", "outputs/checkpoints/constraint-encoder-v1"))
    epochs = int(cfg.get("epochs", 1))
    batch_size = int(cfg.get("batch_size", 16))
    max_seq_len = int(cfg.get("max_seq_len", 128))

    rows = read_jsonl(Path(train_file))
    examples = build_examples(rows)
    if not examples:
        raise RuntimeError(f"No valid training examples from {train_file}")

    transformer = models.Transformer(base_model, max_seq_length=max_seq_len)
    pooling = models.Pooling(transformer.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[transformer, pooling])

    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    warmup_steps = int(len(loader) * epochs * 0.1)
    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=True,
    )

    print(f"Training complete. Saved model to: {output_dir}")


if __name__ == "__main__":
    main()
