from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import PROCESSED_DIR, REPORTS_DIR, ensure_project_dirs, load_sentence_encoder, read_jsonl
from metrics import ccr_at_k, ndcg_at_k, recall_at_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Recall/NDCG/CCR for vanilla and dual retrieval.")
    parser.add_argument(
        "--benchmark-file",
        type=str,
        default=str(PROCESSED_DIR / "retrieval_benchmark_v1.jsonl"),
    )
    parser.add_argument(
        "--corpus-file",
        type=str,
        default=str(PROCESSED_DIR / "retrieval_corpus_v1.jsonl"),
    )
    parser.add_argument("--topic-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--constraint-model", type=str, default="outputs/checkpoints/constraint-encoder-v1")
    parser.add_argument("--retrieve-k", type=int, default=100)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument(
        "--report-file",
        type=str,
        default=str(REPORTS_DIR / "retrieval_metrics_report.json"),
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    benchmark = read_jsonl(Path(args.benchmark_file))
    corpus = read_jsonl(Path(args.corpus_file))
    if not benchmark:
        raise RuntimeError(f"Empty benchmark: {args.benchmark_file}")
    if not corpus:
        raise RuntimeError(f"Empty corpus: {args.corpus_file}")

    doc_ids = [d["doc_id"] for d in corpus]
    doc_texts = [d["text"] for d in corpus]

    topic_model = load_sentence_encoder(args.topic_model)
    constraint_model = load_sentence_encoder(args.constraint_model)

    topic_doc_emb = topic_model.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    constraint_doc_emb = constraint_model.encode(
        doc_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    )

    per_query = []
    cat_stats: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    overall: dict[str, list[float]] = defaultdict(list)

    for item in benchmark:
        query = item["query"]
        category = item.get("category", "unknown")
        topical_relevant = item["topical_relevant_doc_ids"]
        constraint_positive = item["constraint_satisfying_doc_ids"]
        graded_rel = {k: float(v) for k, v in item["graded_relevance"].items()}

        q_topic = topic_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        q_constraint = constraint_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        topic_scores = np.dot(topic_doc_emb, q_topic)
        constraint_scores = np.dot(constraint_doc_emb, q_constraint)

        vanilla_idx = np.argsort(-topic_scores)
        vanilla_rank = [doc_ids[i] for i in vanilla_idx]

        retrieve_idx = vanilla_idx[: args.retrieve_k]
        final_scores = args.alpha * topic_scores[retrieve_idx] + (1.0 - args.alpha) * constraint_scores[retrieve_idx]
        keep_local = [j for j, i in enumerate(retrieve_idx) if float(constraint_scores[i]) >= args.tau]
        if keep_local:
            kept_idx = retrieve_idx[keep_local]
            kept_scores = final_scores[keep_local]
            order = np.argsort(-kept_scores)
            dual_idx = kept_idx[order]
        else:
            dual_idx = retrieve_idx
        dual_rank = [doc_ids[i] for i in dual_idx] + [doc_ids[i] for i in vanilla_idx if i not in set(dual_idx)]

        top10_v = vanilla_rank[:10]
        top10_d = dual_rank[:10]
        row = {
            "query_id": item["query_id"],
            "query": query,
            "category": category,
            "constraint_satisfying_doc_ids": constraint_positive,
            "vanilla": {
                "recall@10": recall_at_k(vanilla_rank, topical_relevant, 10),
                "recall@100": recall_at_k(vanilla_rank, topical_relevant, 100),
                "ndcg@10": ndcg_at_k(vanilla_rank, graded_rel, 10),
                "ccr@10": ccr_at_k(vanilla_rank, constraint_positive, 10),
                "top10_doc_ids": top10_v,
            },
            "dual": {
                "recall@10": recall_at_k(dual_rank, topical_relevant, 10),
                "recall@100": recall_at_k(dual_rank, topical_relevant, 100),
                "ndcg@10": ndcg_at_k(dual_rank, graded_rel, 10),
                "ccr@10": ccr_at_k(dual_rank, constraint_positive, 10),
                "top10_doc_ids": top10_d,
            },
        }
        per_query.append(row)

        for mode in ("vanilla", "dual"):
            for metric, value in row[mode].items():
                overall[f"{mode}_{metric}"].append(value)
                cat_stats[category][f"{mode}_{metric}"].append(value)

    by_category = []
    for category, vals in sorted(cat_stats.items()):
        by_category.append(
            {
                "category": category,
                "count": len(vals["vanilla_ccr@10"]),
                "vanilla": {
                    "recall@10": mean(vals["vanilla_recall@10"]),
                    "recall@100": mean(vals["vanilla_recall@100"]),
                    "ndcg@10": mean(vals["vanilla_ndcg@10"]),
                    "ccr@10": mean(vals["vanilla_ccr@10"]),
                },
                "dual": {
                    "recall@10": mean(vals["dual_recall@10"]),
                    "recall@100": mean(vals["dual_recall@100"]),
                    "ndcg@10": mean(vals["dual_ndcg@10"]),
                    "ccr@10": mean(vals["dual_ccr@10"]),
                },
            }
        )

    report = {
        "benchmark_file": args.benchmark_file,
        "corpus_file": args.corpus_file,
        "topic_model": args.topic_model,
        "constraint_model": args.constraint_model,
        "retrieve_k": args.retrieve_k,
        "final_k": args.final_k,
        "alpha": args.alpha,
        "tau": args.tau,
        "overall": {
            "vanilla": {
                "recall@10": mean(overall["vanilla_recall@10"]),
                "recall@100": mean(overall["vanilla_recall@100"]),
                "ndcg@10": mean(overall["vanilla_ndcg@10"]),
                "ccr@10": mean(overall["vanilla_ccr@10"]),
            },
            "dual": {
                "recall@10": mean(overall["dual_recall@10"]),
                "recall@100": mean(overall["dual_recall@100"]),
                "ndcg@10": mean(overall["dual_ndcg@10"]),
                "ccr@10": mean(overall["dual_ccr@10"]),
            },
        },
        "improvement": {
            "recall@10_abs": mean(overall["dual_recall@10"]) - mean(overall["vanilla_recall@10"]),
            "recall@100_abs": mean(overall["dual_recall@100"]) - mean(overall["vanilla_recall@100"]),
            "ndcg@10_abs": mean(overall["dual_ndcg@10"]) - mean(overall["vanilla_ndcg@10"]),
            "ccr@10_abs": mean(overall["dual_ccr@10"]) - mean(overall["vanilla_ccr@10"]),
        },
        "by_category": by_category,
        "per_query": per_query,
    }

    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2, ensure_ascii=False))
    print(json.dumps(report["improvement"], indent=2, ensure_ascii=False))
    print(f"Saved report to: {out}")


if __name__ == "__main__":
    main()

