from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_BASE_MODEL = PROJECT_ROOT / "models" / "sentence-transformers_all-MiniLM-L6-v2"

trainer_config = {
    "train_file": str(PROJECT_ROOT / "data" / "processed" / "train_triplets.jsonl"),
    "base_model": str(LOCAL_BASE_MODEL),
    "output_dir": str(PROJECT_ROOT / "outputs" / "checkpoints" / "constraint-encoder-v1"),
    "epochs": 1,
    "batch_size": 16,
    "max_seq_len": 128,
    "seed": 42,
}
