from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder

from common import PROCESSED_DIR, REPORTS_DIR, ensure_project_dirs, load_sentence_encoder, read_jsonl


TOKEN_RE = re.compile(r"[a-zA-Z0-9$]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    idx = min(len(arr) - 1, int(math.ceil((p / 100.0) * len(arr)) - 1))
    return arr[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latency benchmark for vanilla/dual/cross-encoder retrieval.")
    parser.add_argument("--benchmark-file", type=str, default=str(PROCESSED_DIR / "retrieval_benchmark_v1.jsonl"))
    parser.add_argument("--corpus-file", type=str, default=str(PROCESSED_DIR / "retrieval_corpus_v1.jsonl"))
    parser.add_argument("--topic-model", type=str, default="/data/xingkun/local_model/Llama-3.2-3B-Instruct")
    parser.add_argument("--constraint-model", type=str, default="outputs/checkpoints/constraint-encoder-v1")
    parser.add_argument("--cross-encoder-model", type=str, default="")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--report-file", type=str, default=str(REPORTS_DIR / "latency_benchmark_report.json"))
    return parser.parse_args()


def bm25_prepare(corpus_texts: list[str]):
    docs = [tokenize(t) for t in corpus_texts]
    tf = [dict() for _ in docs]
    df = {}
    lengths = []
    for i, tokens in enumerate(docs):
        lengths.append(len(tokens))
        local = {}
        for t in tokens:
            local[t] = local.get(t, 0) + 1
        tf[i] = local
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    avgdl = sum(lengths) / len(lengths) if lengths else 0.0
    return docs, tf, df, lengths, avgdl


def bm25_scores(query: str, tf, df, lengths, avgdl, k1=1.5, b=0.75):
    q_tokens = tokenize(query)
    N = len(tf)
    scores = []
    for i, local_tf in enumerate(tf):
        dl = lengths[i]
        s = 0.0
        for t in q_tokens:
            f = local_tf.get(t, 0)
            if f == 0:
                continue
            n = df.get(t, 0)
            idf = math.log(1 + (N - n + 0.5) / (n + 0.5))
            denom = f + k1 * (1 - b + b * dl / (avgdl + 1e-12))
            s += idf * (f * (k1 + 1) / (denom + 1e-12))
        scores.append(s)
    return scores


def summarize(values: list[float]) -> dict:
    return {
        "mean_ms": float(np.mean(values)) if values else 0.0,
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "count": len(values),
    }


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    benchmark = read_jsonl(Path(args.benchmark_file))
    corpus = read_jsonl(Path(args.corpus_file))
    if not benchmark or not corpus:
        raise RuntimeError("Benchmark or corpus is empty.")

    benchmark = benchmark[: args.max_queries]
    doc_texts = [d["text"] for d in corpus]

    topic = load_sentence_encoder(args.topic_model)
    constraint = load_sentence_encoder(args.constraint_model)
    cross = CrossEncoder(args.cross_encoder_model) if args.cross_encoder_model else None

    topic_doc_emb = topic.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    constraint_doc_emb = constraint.encode(
        doc_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    )
    _, tf, df, lengths, avgdl = bm25_prepare(doc_texts)

    vanilla_times = []
    dual_times = []
    bm25_times = []
    cross_times = []

    for item in benchmark:
        q = item["query"]

        t0 = time.perf_counter()
        q_topic = topic.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
        _ = topic_doc_emb @ q_topic
        vanilla_times.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        q_topic_dual = topic.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
        q_con = constraint.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
        topic_scores = topic_doc_emb @ q_topic_dual
        con_scores = constraint_doc_emb @ q_con
        idx = np.argsort(-topic_scores)[: args.candidate_k]
        final_scores = args.alpha * topic_scores[idx] + (1.0 - args.alpha) * con_scores[idx]
        keep = [j for j, i in enumerate(idx) if con_scores[i] >= args.tau]
        if keep:
            _ = idx[np.argsort(-final_scores[keep])]
        dual_times.append((time.perf_counter() - t1) * 1000.0)

        t2 = time.perf_counter()
        _ = bm25_scores(q, tf, df, lengths, avgdl)
        bm25_times.append((time.perf_counter() - t2) * 1000.0)

        if cross is not None:
            t3 = time.perf_counter()
            cand_idx = np.argsort(-topic_scores)[: args.candidate_k]
            pairs = [[q, doc_texts[i]] for i in cand_idx]
            _ = cross.predict(pairs, show_progress_bar=False)
            cross_times.append((time.perf_counter() - t3) * 1000.0)

    report = {
        "num_queries": len(benchmark),
        "topic_model": args.topic_model,
        "constraint_model": args.constraint_model,
        "cross_encoder_model": args.cross_encoder_model or None,
        "candidate_k": args.candidate_k,
        "latency": {
            "vanilla": summarize(vanilla_times),
            "dual": summarize(dual_times),
            "bm25": summarize(bm25_times),
            "cross_encoder": summarize(cross_times) if cross_times else None,
        },
    }

    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["latency"], indent=2, ensure_ascii=False))
    print(f"Saved report to: {out}")


if __name__ == "__main__":
    main()

