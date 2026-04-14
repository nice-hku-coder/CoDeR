from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

trainer_config = {
    "train_file": str(PROJECT_ROOT / "data" / "processed" / "train_triplets.jsonl"),
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",
    "output_dir": str(PROJECT_ROOT / "outputs" / "checkpoints" / "constraint-encoder-v1"),
    "epochs": 1,
    "batch_size": 16,
    "max_seq_len": 128,
    "seed": 42,
}
