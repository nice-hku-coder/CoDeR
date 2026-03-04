from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import PROCESSED_DIR, REPORTS_DIR, ensure_project_dirs, load_sentence_encoder, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CCR for vanilla retrieval vs dual-view retrieval.")
    parser.add_argument("--eval-file", type=str, default=str(PROCESSED_DIR / "smoke_eval.jsonl"))
    parser.add_argument("--topic-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--constraint-model", type=str, default="outputs/checkpoints/constraint-encoder-v1")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--report-file", type=str, default=str(REPORTS_DIR / "rag_eval_report.json"))
    return parser.parse_args()


def encode_scores(model, query: str, docs: list[str]) -> np.ndarray:
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    d = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    return np.dot(d, q[0])


def ccr_at_k(labels: list[int], rank: np.ndarray, k: int) -> float:
    top = rank[:k]
    if len(top) == 0:
        return 0.0
    return float(sum(labels[i] for i in top) / len(top))


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    rows = read_jsonl(Path(args.eval_file))
    if not rows:
        raise RuntimeError(f"Empty eval file: {args.eval_file}. Run build_triplets.py first.")

    topic_model = load_sentence_encoder(args.topic_model)
    constraint_model = load_sentence_encoder(args.constraint_model)

    vanilla_ccr = []
    dual_ccr = []
    per_query = []

    for item in rows:
        query = item["query"]
        docs = [d["text"] for d in item["docs"]]
        labels = [int(d["satisfies"]) for d in item["docs"]]

        t_scores = encode_scores(topic_model, query, docs)
        c_scores = encode_scores(constraint_model, query, docs)
        f_scores = args.alpha * t_scores + (1.0 - args.alpha) * c_scores

        vanilla_rank = np.argsort(-t_scores)
        dual_rank = np.argsort(-f_scores)

        v_ccr = ccr_at_k(labels, vanilla_rank, args.top_k)
        d_ccr = ccr_at_k(labels, dual_rank, args.top_k)
        vanilla_ccr.append(v_ccr)
        dual_ccr.append(d_ccr)

        per_query.append(
            {
                "query": query,
                "vanilla_ccr": v_ccr,
                "dual_ccr": d_ccr,
                "improvement": d_ccr - v_ccr,
            }
        )

    report = {
        "eval_file": args.eval_file,
        "top_k": args.top_k,
        "alpha": args.alpha,
        "avg_vanilla_ccr": float(np.mean(vanilla_ccr) if vanilla_ccr else 0.0),
        "avg_dual_ccr": float(np.mean(dual_ccr) if dual_ccr else 0.0),
        "avg_improvement": float(
            (np.mean(dual_ccr) - np.mean(vanilla_ccr)) if vanilla_ccr and dual_ccr else 0.0
        ),
        "per_query": per_query,
    }

    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved report to: {out}")


if __name__ == "__main__":
    main()
