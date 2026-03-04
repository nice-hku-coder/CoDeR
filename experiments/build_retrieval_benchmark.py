from __future__ import annotations

import argparse
from pathlib import Path

from common import PROCESSED_DIR, ensure_project_dirs, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified retrieval benchmark and corpus.")
    parser.add_argument(
        "--source-eval-file",
        type=str,
        default=str(PROCESSED_DIR / "constraint_benchmark_v1.jsonl"),
    )
    parser.add_argument(
        "--output-corpus-file",
        type=str,
        default=str(PROCESSED_DIR / "retrieval_corpus_v1.jsonl"),
    )
    parser.add_argument(
        "--output-benchmark-file",
        type=str,
        default=str(PROCESSED_DIR / "retrieval_benchmark_v1.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    rows = read_jsonl(Path(args.source_eval_file))
    if not rows:
        raise RuntimeError(f"Empty source eval file: {args.source_eval_file}")

    # Global corpus deduplicated by text
    text_to_id: dict[str, str] = {}
    corpus_rows: list[dict] = []
    for item in rows:
        for doc in item["docs"]:
            text = doc["text"].strip()
            if text in text_to_id:
                continue
            doc_id = f"doc-{len(corpus_rows)}"
            text_to_id[text] = doc_id
            corpus_rows.append({"doc_id": doc_id, "text": text})

    benchmark_rows: list[dict] = []
    for qid, item in enumerate(rows):
        query = item["query"]
        category = item.get("category", "unknown")
        topical_ids: list[str] = []
        constraint_ids: list[str] = []
        graded_relevance: dict[str, float] = {}

        for doc in item["docs"]:
            doc_id = text_to_id[doc["text"].strip()]
            topical_ids.append(doc_id)
            # topical relevance is binary 1 for in-query docs
            graded_relevance[doc_id] = 1.0
            if int(doc["satisfies"]) == 1:
                constraint_ids.append(doc_id)

        benchmark_rows.append(
            {
                "query_id": f"q-{qid}",
                "query": query,
                "category": category,
                "topical_relevant_doc_ids": topical_ids,
                "constraint_satisfying_doc_ids": constraint_ids,
                "graded_relevance": graded_relevance,
            }
        )

    corpus_out = Path(args.output_corpus_file)
    bench_out = Path(args.output_benchmark_file)
    write_jsonl(corpus_out, corpus_rows)
    write_jsonl(bench_out, benchmark_rows)

    print(f"Saved corpus: {len(corpus_rows)} -> {corpus_out}")
    print(f"Saved benchmark: {len(benchmark_rows)} -> {bench_out}")


if __name__ == "__main__":
    main()

