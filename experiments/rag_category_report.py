from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import REPORTS_DIR, ensure_project_dirs, load_sentence_encoder, read_jsonl


def encode_scores(model, query: str, docs: list[str]) -> np.ndarray:
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    d = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    return np.dot(d, q[0])


def ccr_at_k(labels: list[int], rank: np.ndarray, k: int) -> float:
    top = rank[:k]
    if len(top) == 0:
        return 0.0
    return float(sum(labels[i] for i in top) / len(top))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Category-level CCR report for vanilla vs dual retrieval.")
    parser.add_argument("--eval-file", type=str, required=True)
    parser.add_argument("--topic-model", type=str, required=True)
    parser.add_argument("--constraint-model", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=0, help="0 uses all queries")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-file",
        type=str,
        default=str(REPORTS_DIR / "rag_category_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    rows = read_jsonl(Path(args.eval_file))
    if not rows:
        raise RuntimeError(f"Empty eval file: {args.eval_file}")
    if args.max_queries > 0 and len(rows) > args.max_queries:
        rnd = random.Random(args.seed)
        rows = rnd.sample(rows, args.max_queries)

    topic_model = load_sentence_encoder(args.topic_model)
    constraint_model = load_sentence_encoder(args.constraint_model)

    by_category = defaultdict(lambda: {"vanilla": [], "dual": [], "count": 0})
    overall_v = []
    overall_d = []

    for item in rows:
        query = item["query"]
        category = item.get("category", "unknown")
        docs = [d["text"] for d in item["docs"]]
        labels = [int(d["satisfies"]) for d in item["docs"]]

        t_scores = encode_scores(topic_model, query, docs)
        c_scores = encode_scores(constraint_model, query, docs)
        final_scores = args.alpha * t_scores + (1.0 - args.alpha) * c_scores

        vanilla_rank = np.argsort(-t_scores)

        keep_idx = [i for i, cs in enumerate(c_scores) if cs >= args.tau]
        if keep_idx:
            dual_rank = np.array(sorted(keep_idx, key=lambda i: float(final_scores[i]), reverse=True), dtype=np.int32)
        else:
            dual_rank = vanilla_rank

        v = ccr_at_k(labels, vanilla_rank, args.top_k)
        d = ccr_at_k(labels, dual_rank, args.top_k)

        by_category[category]["vanilla"].append(v)
        by_category[category]["dual"].append(d)
        by_category[category]["count"] += 1
        overall_v.append(v)
        overall_d.append(d)

    categories = []
    for cat, values in sorted(by_category.items()):
        v_avg = float(np.mean(values["vanilla"])) if values["vanilla"] else 0.0
        d_avg = float(np.mean(values["dual"])) if values["dual"] else 0.0
        categories.append(
            {
                "category": cat,
                "count": values["count"],
                "vanilla_ccr": v_avg,
                "dual_ccr": d_avg,
                "improvement": d_avg - v_avg,
            }
        )

    report = {
        "eval_file": args.eval_file,
        "num_queries": len(rows),
        "topic_model": args.topic_model,
        "constraint_model": args.constraint_model,
        "alpha": args.alpha,
        "tau": args.tau,
        "top_k": args.top_k,
        "overall": {
            "vanilla_ccr": float(np.mean(overall_v) if overall_v else 0.0),
            "dual_ccr": float(np.mean(overall_d) if overall_d else 0.0),
            "improvement": float((np.mean(overall_d) - np.mean(overall_v)) if overall_v and overall_d else 0.0),
        },
        "by_category": categories,
    }

    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved report to: {out}")


if __name__ == "__main__":
    main()
