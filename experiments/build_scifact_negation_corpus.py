from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import PROCESSED_DIR, ensure_project_dirs, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SciFact negation corpus subset.")
    parser.add_argument(
        "--benchmark-file",
        type=str,
        default=str(PROCESSED_DIR / "scifact_negation_benchmark_v1.jsonl"),
    )
    parser.add_argument(
        "--corpus-file",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "beir" / "scifact" / "corpus.jsonl"),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(PROCESSED_DIR / "scifact_negation_corpus_v1.jsonl"),
    )
    return parser.parse_args()


def read_jsonl_utf8_sig(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_corpus_text(doc: dict) -> str:
    title = str(doc.get("title", "")).strip()
    text = str(doc.get("text", "")).strip()
    if title and text:
        return f"{title} {text}"
    return title or text


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    benchmark_rows = read_jsonl_utf8_sig(Path(args.benchmark_file))
    corpus_rows = read_jsonl_utf8_sig(Path(args.corpus_file))
    if not benchmark_rows:
        raise RuntimeError(f"Empty benchmark file: {args.benchmark_file}")
    if not corpus_rows:
        raise RuntimeError(f"Empty corpus file: {args.corpus_file}")

    needed_ids: list[str] = []
    seen_ids: set[str] = set()
    for item in benchmark_rows:
        for doc_id in item.get("topical_relevant_doc_ids", []):
            doc_id = str(doc_id)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                needed_ids.append(doc_id)
        for doc_id in item.get("constraint_satisfying_doc_ids", []):
            doc_id = str(doc_id)
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                needed_ids.append(doc_id)

    corpus_lookup = {str(doc["_id"]): doc for doc in corpus_rows}
    output_rows: list[dict] = []
    missing_ids: list[str] = []
    for doc_id in needed_ids:
        doc = corpus_lookup.get(doc_id)
        if doc is None:
            missing_ids.append(doc_id)
            continue
        output_rows.append({"doc_id": doc_id, "text": build_corpus_text(doc)})

    if missing_ids:
        raise RuntimeError(f"Missing {len(missing_ids)} corpus ids: {missing_ids[:10]}")

    out = Path(args.output_file)
    write_jsonl(out, output_rows)
    print(f"Saved corpus rows: {len(output_rows)} -> {out}")


if __name__ == "__main__":
    main()