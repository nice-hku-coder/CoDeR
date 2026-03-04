from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from common import PROCESSED_DIR, ensure_project_dirs, set_seed, write_jsonl


@dataclass
class Entity:
    name: str
    good: str
    bad: str
    neutral: str


HOTELS = [
    Entity("hotel", "clean", "dirty", "close to downtown"),
    Entity("room", "quiet", "noisy", "has free wifi"),
    Entity("resort", "family friendly", "unsafe for kids", "has a swimming pool"),
]

FOODS = [
    Entity("meal", "peanut-free", "contains peanuts", "served warm"),
    Entity("dish", "gluten-free", "contains gluten", "has rich flavor"),
    Entity("menu", "low sugar", "high sugar", "includes dessert"),
]

PHONES = [
    ("under 100 dollars", "$89", "$699"),
    ("under 300 dollars", "$249", "$899"),
    ("under 500 dollars", "$399", "$1099"),
]


def build_negation_samples(target_n: int, rng: random.Random) -> list[dict]:
    rows = []
    for _ in range(target_n):
        e = rng.choice(HOTELS)
        query = f"Find {e.name}s that are not {e.bad}"
        docs = [
            {"text": f"This {e.name} is {e.good} and well maintained.", "satisfies": 1},
            {"text": f"Guests report this {e.name} is {e.bad}.", "satisfies": 0},
            {"text": f"The {e.name} is {e.neutral}.", "satisfies": 1},
            {"text": f"Many reviews complain the {e.name} feels {e.bad}.", "satisfies": 0},
            {"text": f"A {e.good} {e.name} with friendly staff.", "satisfies": 1},
            {"text": f"This place is often described as {e.bad}.", "satisfies": 0},
        ]
        rows.append({"query": query, "docs": docs, "category": "negation"})
    return rows


def build_exclusion_samples(target_n: int, rng: random.Random) -> list[dict]:
    rows = []
    for _ in range(target_n):
        e = rng.choice(FOODS)
        query = f"Suggest {e.name}s without items that {e.bad}"
        docs = [
            {"text": f"This {e.name} is {e.good}.", "satisfies": 1},
            {"text": f"The {e.name} {e.bad}.", "satisfies": 0},
            {"text": f"Our chef prepared a {e.good} option.", "satisfies": 1},
            {"text": f"Warning: this choice {e.bad}.", "satisfies": 0},
            {"text": f"The {e.name} is {e.neutral}.", "satisfies": 1},
            {"text": f"Popular pick that {e.bad}.", "satisfies": 0},
        ]
        rows.append({"query": query, "docs": docs, "category": "exclusion"})
    return rows


def build_numeric_samples(target_n: int, rng: random.Random) -> list[dict]:
    rows = []
    for _ in range(target_n):
        constraint, cheap_price, expensive_price = rng.choice(PHONES)
        query = f"Recommend phones {constraint}"
        docs = [
            {"text": f"Budget phone priced at {cheap_price} with 4GB RAM.", "satisfies": 1},
            {"text": f"Premium phone priced at {expensive_price}.", "satisfies": 0},
            {"text": f"Affordable device at {cheap_price} with long battery life.", "satisfies": 1},
            {"text": f"Flagship model at {expensive_price} with OLED display.", "satisfies": 0},
            {"text": f"Entry model costs {cheap_price}.", "satisfies": 1},
            {"text": f"Luxury edition costs {expensive_price}.", "satisfies": 0},
        ]
        rows.append({"query": query, "docs": docs, "category": "numeric"})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a larger synthetic benchmark for constraint-aware retrieval.")
    parser.add_argument("--output-file", type=str, default=str(PROCESSED_DIR / "constraint_benchmark_v1.jsonl"))
    parser.add_argument("--num-negation", type=int, default=80)
    parser.add_argument("--num-exclusion", type=int, default=80)
    parser.add_argument("--num-numeric", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    set_seed(args.seed)
    rng = random.Random(args.seed)

    rows = []
    rows.extend(build_negation_samples(args.num_negation, rng))
    rows.extend(build_exclusion_samples(args.num_exclusion, rng))
    rows.extend(build_numeric_samples(args.num_numeric, rng))
    rng.shuffle(rows)

    out = Path(args.output_file)
    write_jsonl(out, rows)
    print(f"Saved benchmark rows: {len(rows)} -> {out}")


if __name__ == "__main__":
    main()
