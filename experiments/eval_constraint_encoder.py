from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import (
    PROCESSED_DIR,
    REPORTS_DIR,
    ensure_project_dirs,
    load_sentence_encoder,
    pairwise_accuracy,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate constraint encoder against baseline embedding model.")
    parser.add_argument("--eval-file", type=str, default=str(PROCESSED_DIR / "val_triplets.jsonl"))
    parser.add_argument("--constraint-model", type=str, default="outputs/checkpoints/constraint-encoder-v1")
    parser.add_argument("--baseline-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--report-file", type=str, default=str(REPORTS_DIR / "constraint_eval_report.json"))
    return parser.parse_args()


def score_model(model, rows: list[dict]) -> tuple[list[float], list[float]]:
    queries = [r["query"] for r in rows]
    positives = [r["positive"] for r in rows]
    negatives = [r["hard_negative"] for r in rows]

    q_emb = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    p_emb = model.encode(positives, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    n_emb = model.encode(negatives, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

    pos_scores = np.sum(q_emb * p_emb, axis=1).tolist()
    neg_scores = np.sum(q_emb * n_emb, axis=1).tolist()
    return pos_scores, neg_scores


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    rows = read_jsonl(Path(args.eval_file))
    if not rows:
        raise RuntimeError(f"No eval samples loaded from {args.eval_file}")

    baseline = load_sentence_encoder(args.baseline_model)
    base_pos, base_neg = score_model(baseline, rows)
    base_metrics = pairwise_accuracy(base_pos, base_neg)

    constraint = load_sentence_encoder(args.constraint_model)
    con_pos, con_neg = score_model(constraint, rows)
    con_metrics = pairwise_accuracy(con_pos, con_neg)

    report = {
        "eval_file": args.eval_file,
        "baseline_model": args.baseline_model,
        "constraint_model": args.constraint_model,
        "baseline": base_metrics.to_dict(),
        "constraint": con_metrics.to_dict(),
        "improvement": {
            "accuracy_abs": con_metrics.accuracy - base_metrics.accuracy,
            "avg_margin_abs": con_metrics.avg_margin - base_metrics.avg_margin,
        },
    }

    out_path = Path(args.report_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved report to: {out_path}")


if __name__ == "__main__":
    main()
