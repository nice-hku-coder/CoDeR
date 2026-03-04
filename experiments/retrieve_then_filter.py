from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import PROCESSED_DIR, load_sentence_encoder, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-view retrieval: topic retrieve then constraint rerank/filter.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--corpus-file", type=str, default=str(PROCESSED_DIR / "demo_corpus.jsonl"))
    parser.add_argument("--topic-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--constraint-model", type=str, default="outputs/checkpoints/constraint-encoder-v1")
    parser.add_argument("--top-k-retrieve", type=int, default=20)
    parser.add_argument("--top-k-final", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--tau", type=float, default=None)
    return parser.parse_args()


def cosine_scores(model, query: str, docs: list[str]) -> np.ndarray:
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    d = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    return np.dot(d, q[0])


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.corpus_file))
    if not rows:
        raise RuntimeError(f"Empty corpus: {args.corpus_file}. Run build_triplets.py first.")

    docs = [r["text"] for r in rows]
    ids = [r.get("id", f"doc-{i}") for i, r in enumerate(rows)]

    topic_model = load_sentence_encoder(args.topic_model)
    topic_scores = cosine_scores(topic_model, args.query, docs)

    retrieve_idx = np.argsort(-topic_scores)[: args.top_k_retrieve]
    cand_docs = [docs[i] for i in retrieve_idx]
    cand_ids = [ids[i] for i in retrieve_idx]
    cand_topic = topic_scores[retrieve_idx]

    constraint_model = load_sentence_encoder(args.constraint_model)
    constraint_scores = cosine_scores(constraint_model, args.query, cand_docs)

    if args.tau is not None:
        keep = constraint_scores >= args.tau
        cand_docs = [d for d, k in zip(cand_docs, keep) if k]
        cand_ids = [d for d, k in zip(cand_ids, keep) if k]
        cand_topic = np.array([s for s, k in zip(cand_topic, keep) if k], dtype=np.float32)
        constraint_scores = np.array([s for s, k in zip(constraint_scores, keep) if k], dtype=np.float32)

    final_scores = args.alpha * cand_topic + (1.0 - args.alpha) * constraint_scores
    rank = np.argsort(-final_scores)[: args.top_k_final]

    print(f"Query: {args.query}\n")
    print("Top results after dual-view rerank:")
    for i, idx in enumerate(rank, start=1):
        print(f"[{i}] {cand_ids[idx]}")
        print(f"  final={final_scores[idx]:.4f} topic={cand_topic[idx]:.4f} constraint={constraint_scores[idx]:.4f}")
        print(f"  text: {cand_docs[idx]}")


if __name__ == "__main__":
    main()
