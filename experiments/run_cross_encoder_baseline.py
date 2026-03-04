from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from sentence_transformers import CrossEncoder

from common import PROCESSED_DIR, REPORTS_DIR, ensure_project_dirs, load_sentence_encoder, read_jsonl
from metrics import ccr_at_k, ndcg_at_k, recall_at_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cross-Encoder reranker baseline.")
    parser.add_argument("--benchmark-file", type=str, default=str(PROCESSED_DIR / "retrieval_benchmark_v1.jsonl"))
    parser.add_argument("--corpus-file", type=str, default=str(PROCESSED_DIR / "retrieval_corpus_v1.jsonl"))
    parser.add_argument("--topic-model", type=str, default="/data/xingkun/local_model/Llama-3.2-3B-Instruct")
    parser.add_argument("--cross-encoder-model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--report-file", type=str, default=str(REPORTS_DIR / "cross_encoder_metrics_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    benchmark = read_jsonl(Path(args.benchmark_file))
    corpus = read_jsonl(Path(args.corpus_file))
    if not benchmark or not corpus:
        raise RuntimeError("Benchmark or corpus is empty.")

    topic = load_sentence_encoder(args.topic_model)
    cross = CrossEncoder(args.cross_encoder_model)

    doc_ids = [d["doc_id"] for d in corpus]
    doc_texts = [d["text"] for d in corpus]
    topic_doc_emb = topic.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

    overall = defaultdict(list)
    per_query = []
    latencies = []

    for item in benchmark:
        query = item["query"]
        topical = item["topical_relevant_doc_ids"]
        constraint = item["constraint_satisfying_doc_ids"]
        graded = {k: float(v) for k, v in item["graded_relevance"].items()}

        t0 = time.perf_counter()
        q_emb = topic.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        topic_scores = topic_doc_emb @ q_emb
        cand_idx = topic_scores.argsort()[::-1][: args.candidate_k]
        pairs = [[query, doc_texts[i]] for i in cand_idx]
        ce_scores = cross.predict(pairs, show_progress_bar=False)
        if hasattr(ce_scores, "ndim") and ce_scores.ndim == 2:
            entailment_idx = None
            id2label = getattr(cross.model.config, "id2label", {}) or {}
            for idx, label in id2label.items():
                if "entail" in str(label).lower():
                    entailment_idx = int(idx)
                    break
            if entailment_idx is None:
                entailment_idx = ce_scores.shape[1] - 1
            ce_scores = ce_scores[:, entailment_idx]

        order = ce_scores.argsort()[::-1]
        ranked_idx = [int(cand_idx[int(i)]) for i in order]
        # append remaining docs by topic rank so recall@100 remains computable
        ranked_set = set(ranked_idx)
        remaining = [int(i) for i in topic_scores.argsort()[::-1] if int(i) not in ranked_set]
        final_idx = ranked_idx + remaining
        latencies.append((time.perf_counter() - t0) * 1000.0)

        rank = [doc_ids[i] for i in final_idx]
        row = {
            "query_id": item["query_id"],
            "query": query,
            "category": item.get("category", "unknown"),
            "recall@10": recall_at_k(rank, topical, 10),
            "recall@100": recall_at_k(rank, topical, 100),
            "ndcg@10": ndcg_at_k(rank, graded, 10),
            "ccr@10": ccr_at_k(rank, constraint, 10),
            "top10_doc_ids": rank[:10],
            "constraint_satisfying_doc_ids": constraint,
        }
        per_query.append(row)
        for k in ("recall@10", "recall@100", "ndcg@10", "ccr@10"):
            overall[k].append(row[k])

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0.0
    p95 = latencies_sorted[min(len(latencies_sorted) - 1, int(len(latencies_sorted) * 0.95))] if latencies_sorted else 0.0

    report = {
        "benchmark_file": args.benchmark_file,
        "corpus_file": args.corpus_file,
        "topic_model": args.topic_model,
        "cross_encoder_model": args.cross_encoder_model,
        "candidate_k": args.candidate_k,
        "overall": {k: (sum(v) / len(v) if v else 0.0) for k, v in overall.items()},
        "latency_ms": {
            "p50": p50,
            "p95": p95,
            "mean": (sum(latencies) / len(latencies) if latencies else 0.0),
        },
        "per_query": per_query,
    }

    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2, ensure_ascii=False))
    print(json.dumps(report["latency_ms"], indent=2, ensure_ascii=False))
    print(f"Saved report to: {out}")


if __name__ == "__main__":
    main()

