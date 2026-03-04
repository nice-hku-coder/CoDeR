from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from common import PROCESSED_DIR, REPORTS_DIR, ensure_project_dirs, read_jsonl
from metrics import ccr_at_k, ndcg_at_k, recall_at_k


TOKEN_RE = re.compile(r"[a-zA-Z0-9$]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if docs else 0.0
        self.tf = [Counter(d) for d in docs]
        self.df = defaultdict(int)
        for doc in docs:
            for t in set(doc):
                self.df[t] += 1
        self.N = len(docs)

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        if self.N == 0:
            return 0.0
        tf = self.tf[doc_idx]
        dl = self.doc_len[doc_idx]
        score = 0.0
        for t in query_tokens:
            f = tf.get(t, 0)
            if f == 0:
                continue
            idf = self.idf(t)
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-12))
            score += idf * (f * (self.k1 + 1) / (denom + 1e-12))
        return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BM25 baseline on retrieval benchmark.")
    parser.add_argument("--benchmark-file", type=str, default=str(PROCESSED_DIR / "retrieval_benchmark_v1.jsonl"))
    parser.add_argument("--corpus-file", type=str, default=str(PROCESSED_DIR / "retrieval_corpus_v1.jsonl"))
    parser.add_argument("--report-file", type=str, default=str(REPORTS_DIR / "bm25_metrics_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    benchmark = read_jsonl(Path(args.benchmark_file))
    corpus = read_jsonl(Path(args.corpus_file))
    if not benchmark or not corpus:
        raise RuntimeError("Benchmark or corpus is empty.")

    doc_ids = [d["doc_id"] for d in corpus]
    tokenized_docs = [tokenize(d["text"]) for d in corpus]
    bm25 = BM25(tokenized_docs)

    overall = defaultdict(list)
    per_query = []

    for item in benchmark:
        q_tokens = tokenize(item["query"])
        scores = [bm25.score(q_tokens, i) for i in range(len(corpus))]
        rank_idx = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)
        rank = [doc_ids[i] for i in rank_idx]

        topical = item["topical_relevant_doc_ids"]
        constraint = item["constraint_satisfying_doc_ids"]
        graded = {k: float(v) for k, v in item["graded_relevance"].items()}

        row = {
            "query_id": item["query_id"],
            "query": item["query"],
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

    report = {
        "benchmark_file": args.benchmark_file,
        "corpus_file": args.corpus_file,
        "model": "BM25",
        "overall": {k: (sum(v) / len(v) if v else 0.0) for k, v in overall.items()},
        "per_query": per_query,
    }

    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2, ensure_ascii=False))
    print(f"Saved report to: {out}")


if __name__ == "__main__":
    main()

