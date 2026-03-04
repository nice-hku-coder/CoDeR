from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

from common import FIGURES_DIR, REPORTS_DIR, ensure_project_dirs, load_sentence_encoder, set_seed


def collect_pairs(dataset_name: str, split: str, max_samples: int) -> tuple[list[str], list[str], list[str]]:
    ds = load_dataset(dataset_name, split=split)
    grouped: dict[str, dict[int, str]] = {}
    premises: list[str] = []
    entailments: list[str] = []
    contradictions: list[str] = []

    for row in ds:
        label = int(row.get("label", -1))
        if label not in (0, 2):
            continue
        premise = row.get("premise", "").strip()
        hypothesis = row.get("hypothesis", "").strip()
        if not premise or not hypothesis:
            continue

        if premise not in grouped:
            grouped[premise] = {}
        grouped[premise][label] = hypothesis

        if 0 in grouped[premise] and 2 in grouped[premise]:
            premises.append(premise)
            entailments.append(grouped[premise][0])
            contradictions.append(grouped[premise][2])
            grouped.pop(premise, None)
            if len(premises) >= max_samples:
                break

    return premises, entailments, contradictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PoC: quantify negation/contradiction confusion in embeddings.")
    parser.add_argument("--dataset", type=str, default="snli")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    set_seed(args.seed)

    premises, entailments, contradictions = collect_pairs(args.dataset, args.split, args.max_samples)
    if not premises:
        raise RuntimeError("No valid NLI pairs found.")

    model = load_sentence_encoder(args.model)
    premise_emb = model.encode(premises, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    ent_emb = model.encode(entailments, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    con_emb = model.encode(contradictions, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

    ent_scores = np.sum(premise_emb * ent_emb, axis=1)
    con_scores = np.sum(premise_emb * con_emb, axis=1)
    margins = ent_scores - con_scores

    report_path = REPORTS_DIR / "poc_negation_gap_summary.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Samples: {len(premises)}\n")
        f.write(f"Mean entailment similarity: {ent_scores.mean():.4f}\n")
        f.write(f"Mean contradiction similarity: {con_scores.mean():.4f}\n")
        f.write(f"Mean margin (ent - contra): {margins.mean():.4f}\n")
        f.write(f"Pairwise accuracy (ent > contra): {(margins > 0).mean():.4f}\n")

    csv_path = REPORTS_DIR / "poc_negation_gap_scores.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["premise", "entailment", "contradiction", "ent_score", "contra_score", "margin"])
        for p, e, c, es, cs, m in zip(premises, entailments, contradictions, ent_scores, con_scores, margins):
            writer.writerow([p, e, c, f"{es:.6f}", f"{cs:.6f}", f"{m:.6f}"])

    fig_path = FIGURES_DIR / "poc_negation_gap_hist.png"
    plt.figure(figsize=(8, 5))
    plt.hist(ent_scores, bins=40, alpha=0.6, label="Premise vs Entailment")
    plt.hist(con_scores, bins=40, alpha=0.6, label="Premise vs Contradiction")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.title("PoC: Similarity Distribution on NLI Pairs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)

    print(f"Saved summary: {report_path}")
    print(f"Saved scores csv: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
