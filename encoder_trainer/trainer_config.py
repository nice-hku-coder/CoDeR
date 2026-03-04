trainer_config = {
    "train_file": "/home/xingkun/CoDeR/data/processed/train_triplets.jsonl",
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",
    "output_dir": "/home/xingkun/CoDeR/outputs/checkpoints/constraint-encoder-v1",
    "epochs": 1,
    "batch_size": 16,
    "max_seq_len": 128,
    "seed": 42,
}
