from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from common import PROCESSED_DIR, ensure_project_dirs, read_jsonl, write_jsonl


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
            for term in set(doc):
                self.df[term] += 1
        self.n_docs = len(docs)

    def idf(self, term: str) -> float:
        doc_freq = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        if self.n_docs == 0:
            return 0.0
        tf = self.tf[doc_idx]
        doc_len = self.doc_len[doc_idx]
        score = 0.0
        norm = self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl + 1e-12))
        for term in query_tokens:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            score += self.idf(term) * (freq * (self.k1 + 1) / (freq + norm + 1e-12))
        return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a negation benchmark from SciFact support annotations.")
    parser.add_argument(
        "--query-file",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "beir" / "scifact" / "query_with_support_negation.jsonl"),
    )
    parser.add_argument(
        "--corpus-file",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "beir" / "scifact" / "corpus.jsonl"),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(PROCESSED_DIR / "scifact_negation_benchmark_v1.jsonl"),
    )
    parser.add_argument("--min-topical-docs", type=int, default=5)
    return parser.parse_args()


def build_corpus_text(doc: dict) -> str:
    title = str(doc.get("title", "")).strip()
    text = str(doc.get("text", "")).strip()
    if title and text:
        return f"{title} {text}"
    return title or text


def unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def read_jsonl_utf8_sig(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    query_rows = read_jsonl_utf8_sig(Path(args.query_file))
    corpus_rows = read_jsonl_utf8_sig(Path(args.corpus_file))
    if not query_rows:
        raise RuntimeError(f"Empty query file: {args.query_file}")
    if not corpus_rows:
        raise RuntimeError(f"Empty corpus file: {args.corpus_file}")

    corpus_ids = [str(doc["_id"]) for doc in corpus_rows]
    corpus_texts = [build_corpus_text(doc) for doc in corpus_rows]
    bm25 = BM25([tokenize(text) for text in corpus_texts])

    benchmark_rows: list[dict] = []

    for item in query_rows:
        query = str(item["query"])
        query_id = str(item["query-id"])

        support_ids: list[str] = []
        topical_ids: list[str] = []
        topical_seen: set[str] = set()

        for doc in item.get("corpus", []):
            corpus_id = str(doc["corpus-id"])
            label = str(doc.get("type", "")).upper()
            if label == "SUPPORT":
                support_ids.append(corpus_id)
            if label in {"SUPPORT", "RELEVANT"} and corpus_id not in topical_seen:
                topical_ids.append(corpus_id)
                topical_seen.add(corpus_id)

        if len(topical_ids) < args.min_topical_docs:
            query_tokens = tokenize(query)
            scores = [bm25.score(query_tokens, idx) for idx in range(len(corpus_rows))]
            ranked_indices = sorted(range(len(corpus_rows)), key=lambda idx: scores[idx], reverse=True)
            fill_needed = args.min_topical_docs - len(topical_ids)
            for idx in ranked_indices:
                candidate_id = corpus_ids[idx]
                if candidate_id in topical_seen:
                    continue
                topical_ids.append(candidate_id)
                topical_seen.add(candidate_id)
                fill_needed -= 1
                if fill_needed == 0:
                    break

        topical_ids = unique_in_order(topical_ids)
        constraint_ids = unique_in_order(support_ids)
        graded_relevance = {doc_id: 1.0 for doc_id in topical_ids}

        benchmark_rows.append(
            {
                "query_id": query_id,
                "query": query,
                "category": "negation",
                "topical_relevant_doc_ids": topical_ids,
                "constraint_satisfying_doc_ids": constraint_ids,
                "graded_relevance": graded_relevance,
            }
        )

    out = Path(args.output_file)
    write_jsonl(out, benchmark_rows)
    print(f"Saved benchmark rows: {len(benchmark_rows)} -> {out}")


if __name__ == "__main__":
    main()