from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List

from datasets import load_dataset

from common import PROCESSED_DIR, ensure_project_dirs, set_seed, write_jsonl


def build_triplets(dataset_name: str, split: str, max_samples: int) -> list[dict]:
    ds = load_dataset(dataset_name, split=split)
    grouped: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    triplets: list[dict] = []

    for row in ds:
        label = int(row.get("label", -1))
        if label not in (0, 2):
            continue
        premise = row.get("premise", "").strip()
        hypothesis = row.get("hypothesis", "").strip()
        if not premise or not hypothesis:
            continue
        grouped[premise][label].append(hypothesis)

        if grouped[premise][0] and grouped[premise][2]:
            triplets.append(
                {
                    "query": premise,
                    "positive": grouped[premise][0][0],
                    "hard_negative": grouped[premise][2][0],
                }
            )
            grouped.pop(premise, None)
            if len(triplets) >= max_samples:
                break
    return triplets


def build_smoke_eval() -> list[dict]:
    return [
        {
            "query": "Find hotels that are not dirty",
            "docs": [
                {"text": "The hotel is clean and hygienic with spotless rooms.", "satisfies": 1},
                {"text": "Guests report dirty bathrooms and poor hygiene.", "satisfies": 0},
                {"text": "The location is near downtown attractions.", "satisfies": 1},
                {"text": "Many reviews complain the room was dirty.", "satisfies": 0},
            ],
        },
        {
            "query": "Meals without peanuts",
            "docs": [
                {"text": "This dish is peanut-free and allergy safe.", "satisfies": 1},
                {"text": "Contains peanuts and sesame.", "satisfies": 0},
                {"text": "Nut-free menu options available.", "satisfies": 1},
                {"text": "Peanut sauce included by default.", "satisfies": 0},
            ],
        },
        {
            "query": "Phones under 100 dollars",
            "docs": [
                {"text": "Budget phone priced at $89 with 4GB RAM.", "satisfies": 1},
                {"text": "Premium model costs $699.", "satisfies": 0},
                {"text": "Entry-level device at $99.", "satisfies": 1},
                {"text": "Flagship phone at $999 with OLED display.", "satisfies": 0},
            ],
        },
    ]


def flatten_demo_corpus(smoke_eval: list[dict]) -> list[dict]:
    rows: list[dict] = []
    seen = set()
    for item in smoke_eval:
        for doc in item["docs"]:
            text = doc["text"]
            if text in seen:
                continue
            seen.add(text)
            rows.append({"id": f"doc-{len(rows)}", "text": text})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NLI triplets and smoke RAG eval data.")
    parser.add_argument("--dataset", type=str, default="snli")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="validation")
    parser.add_argument("--train-max", type=int, default=20000)
    parser.add_argument("--val-max", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    set_seed(args.seed)

    train_triplets = build_triplets(args.dataset, args.train_split, args.train_max)
    val_triplets = build_triplets(args.dataset, args.val_split, args.val_max)

    train_out = PROCESSED_DIR / "train_triplets.jsonl"
    val_out = PROCESSED_DIR / "val_triplets.jsonl"
    write_jsonl(train_out, train_triplets)
    write_jsonl(val_out, val_triplets)

    smoke_eval = build_smoke_eval()
    smoke_out = PROCESSED_DIR / "smoke_eval.jsonl"
    write_jsonl(smoke_out, smoke_eval)

    demo_corpus = flatten_demo_corpus(smoke_eval)
    corpus_out = PROCESSED_DIR / "demo_corpus.jsonl"
    write_jsonl(corpus_out, demo_corpus)

    print(f"Saved train triplets: {len(train_triplets)} -> {train_out}")
    print(f"Saved val triplets: {len(val_triplets)} -> {val_out}")
    print(f"Saved smoke eval set: {len(smoke_eval)} -> {smoke_out}")
    print(f"Saved demo corpus: {len(demo_corpus)} -> {corpus_out}")


if __name__ == "__main__":
    main()
