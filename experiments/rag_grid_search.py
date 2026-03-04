from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from common import PROCESSED_DIR, REPORTS_DIR, ensure_project_dirs, load_sentence_encoder, read_jsonl


def parse_grid(text: str, cast=float) -> list:
    items = [x.strip() for x in text.split(",") if x.strip()]
    return [cast(x) for x in items]


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
    parser = argparse.ArgumentParser(description="Grid search alpha/tau for dual-view RAG CCR.")
    parser.add_argument("--eval-file", type=str, default=str(PROCESSED_DIR / "smoke_eval.jsonl"))
    parser.add_argument("--topic-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--constraint-model", type=str, default="outputs/checkpoints/constraint-encoder-v1")
    parser.add_argument("--alphas", type=str, default="0.0,0.1,0.3,0.5,0.7,0.9,1.0")
    parser.add_argument("--taus", type=str, default="-1.0,0.0,0.2,0.4,0.6")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=0, help="0 means use all queries.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-file", type=str, default=str(REPORTS_DIR / "rag_grid_search_report.json"))
    args = parser.parse_args()

    ensure_project_dirs()
    rows = read_jsonl(Path(args.eval_file))
    if not rows:
        raise RuntimeError(f"Empty eval file: {args.eval_file}")
    if args.max_queries > 0 and len(rows) > args.max_queries:
        rnd = random.Random(args.seed)
        rows = rnd.sample(rows, args.max_queries)

    alphas = parse_grid(args.alphas, float)
    taus = parse_grid(args.taus, float)

    topic_model = load_sentence_encoder(args.topic_model)
    constraint_model = load_sentence_encoder(args.constraint_model)

    prepared = []
    for item in rows:
        query = item["query"]
        docs = [d["text"] for d in item["docs"]]
        labels = [int(d["satisfies"]) for d in item["docs"]]
        t_scores = encode_scores(topic_model, query, docs)
        c_scores = encode_scores(constraint_model, query, docs)
        prepared.append({"query": query, "labels": labels, "t": t_scores, "c": c_scores, "n_docs": len(docs)})

    results = []
    best = None
    for alpha in alphas:
        for tau in taus:
            per_query = []
            for p in prepared:
                labels = p["labels"]
                f_scores = alpha * p["t"] + (1.0 - alpha) * p["c"]

                keep_idx = [i for i, cs in enumerate(p["c"]) if cs >= tau]
                if keep_idx:
                    ranked = sorted(keep_idx, key=lambda i: float(f_scores[i]), reverse=True)
                else:
                    ranked = sorted(range(p["n_docs"]), key=lambda i: float(p["t"][i]), reverse=True)

                ccr = ccr_at_k(labels, np.array(ranked, dtype=np.int32), args.top_k)
                per_query.append({"query": p["query"], "ccr": ccr})

            avg_ccr = float(np.mean([x["ccr"] for x in per_query])) if per_query else 0.0
            row = {"alpha": alpha, "tau": tau, "avg_ccr": avg_ccr, "per_query": per_query}
            results.append(row)
            if best is None or avg_ccr > best["avg_ccr"]:
                best = row

    vanilla_rows = []
    for p in prepared:
        ranked = sorted(range(p["n_docs"]), key=lambda i: float(p["t"][i]), reverse=True)
        vanilla_rows.append(ccr_at_k(p["labels"], np.array(ranked, dtype=np.int32), args.top_k))
    vanilla_avg = float(np.mean(vanilla_rows)) if vanilla_rows else 0.0

    report = {
        "eval_file": args.eval_file,
        "topic_model": args.topic_model,
        "constraint_model": args.constraint_model,
        "top_k": args.top_k,
        "num_queries": len(rows),
        "alphas": alphas,
        "taus": taus,
        "vanilla_avg_ccr": vanilla_avg,
        "best": best,
        "best_improvement": (best["avg_ccr"] - vanilla_avg) if best else 0.0,
        "all_results": sorted(results, key=lambda x: x["avg_ccr"], reverse=True),
    }

    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Vanilla avg CCR@{args.top_k}: {vanilla_avg:.4f}")
    if best:
        print(
            f"Best dual: alpha={best['alpha']:.2f}, tau={best['tau']:.2f}, "
            f"avg_ccr={best['avg_ccr']:.4f}, improvement={best['avg_ccr'] - vanilla_avg:.4f}"
        )
    print(f"Saved report to: {out}")


if __name__ == "__main__":
    main()
