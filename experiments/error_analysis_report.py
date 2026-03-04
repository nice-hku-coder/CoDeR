from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import REPORTS_DIR, ensure_project_dirs, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate failure-case report from retrieval metrics output.")
    parser.add_argument("--retrieval-report", type=str, required=True)
    parser.add_argument("--corpus-file", type=str, required=True)
    parser.add_argument("--min-cases", type=int, default=20)
    parser.add_argument("--output-md", type=str, default=str(REPORTS_DIR / "error_analysis_cases.md"))
    return parser.parse_args()


def classify_failure(category: str, top1_ok: bool, delta_ccr: float) -> str:
    if category == "negation" and not top1_ok:
        return "negation-scope failure"
    if category == "numeric" and delta_ccr < 0:
        return "numeric-threshold failure"
    if category == "exclusion" and not top1_ok:
        return "exclusion-conflict not filtered"
    if delta_ccr < 0:
        return "dual worse than vanilla"
    return "hard residual case"


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    report = json.loads(Path(args.retrieval_report).read_text(encoding="utf-8"))
    corpus = read_jsonl(Path(args.corpus_file))
    id2text = {d["doc_id"]: d["text"] for d in corpus}

    rows = report["per_query"]
    scored = []
    for r in rows:
        cset = set(r["constraint_satisfying_doc_ids"])
        v_ccr = float(r["vanilla"]["ccr@10"])
        d_ccr = float(r["dual"]["ccr@10"])
        delta = d_ccr - v_ccr
        v_top1 = r["vanilla"]["top10_doc_ids"][0] if r["vanilla"]["top10_doc_ids"] else ""
        d_top1 = r["dual"]["top10_doc_ids"][0] if r["dual"]["top10_doc_ids"] else ""
        d_top1_ok = d_top1 in cset
        scored.append(
            {
                "query_id": r["query_id"],
                "query": r["query"],
                "category": r.get("category", "unknown"),
                "vanilla_ccr@10": v_ccr,
                "dual_ccr@10": d_ccr,
                "delta_ccr@10": delta,
                "vanilla_top1": v_top1,
                "dual_top1": d_top1,
                "dual_top1_ok": d_top1_ok,
                "failure_type": classify_failure(r.get("category", "unknown"), d_top1_ok, delta),
            }
        )

    scored.sort(key=lambda x: (x["delta_ccr@10"], x["dual_ccr@10"]))
    selected = scored[: args.min_cases]

    lines = [
        "# Error Analysis Cases",
        "",
        f"- Source report: `{args.retrieval_report}`",
        f"- Number of cases: {len(selected)}",
        "",
    ]
    for i, s in enumerate(selected, start=1):
        lines.append(f"## Case {i}: {s['query_id']}")
        lines.append(f"- Query: {s['query']}")
        lines.append(f"- Category: {s['category']}")
        lines.append(f"- Vanilla CCR@10: {s['vanilla_ccr@10']:.4f}")
        lines.append(f"- Dual CCR@10: {s['dual_ccr@10']:.4f}")
        lines.append(f"- Delta: {s['delta_ccr@10']:.4f}")
        lines.append(f"- Failure type: {s['failure_type']}")
        lines.append(f"- Vanilla top1: `{s['vanilla_top1']}`")
        lines.append(f"  - {id2text.get(s['vanilla_top1'], '(missing)')}")
        lines.append(f"- Dual top1: `{s['dual_top1']}`")
        lines.append(f"  - {id2text.get(s['dual_top1'], '(missing)')}")
        lines.append("")

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved error analysis markdown to: {out}")


if __name__ == "__main__":
    main()

