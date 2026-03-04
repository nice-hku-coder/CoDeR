from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import REPORTS_DIR, ensure_project_dirs, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proxy end-to-end RAG evaluation from retrieval outputs.")
    parser.add_argument("--retrieval-report", type=str, required=True, help="Output of eval_retrieval_metrics.py")
    parser.add_argument("--mode", type=str, default="dual", choices=["dual", "vanilla"])
    parser.add_argument("--report-file", type=str, default=str(REPORTS_DIR / "rag_e2e_proxy_report.json"))
    return parser.parse_args()


def is_success(top10_doc_ids: list[str], constraint_satisfy: set[str]) -> bool:
    if not top10_doc_ids:
        return False
    # Proxy: if top-1 satisfies constraints, we mark answer as faithful.
    return top10_doc_ids[0] in constraint_satisfy


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    report = json.loads(Path(args.retrieval_report).read_text(encoding="utf-8"))
    per_query = report["per_query"]

    by_cat = defaultdict(list)
    overall = []
    negation_flags = []

    for q in per_query:
        constraint_set = set(q["constraint_satisfying_doc_ids"])
        top10 = q[args.mode]["top10_doc_ids"]
        ok = is_success(top10, constraint_set)
        overall.append(1.0 if ok else 0.0)
        cat = q.get("category", "unknown")
        by_cat[cat].append(1.0 if ok else 0.0)
        if cat == "negation":
            negation_flags.append(1.0 if (not ok) else 0.0)

    output = {
        "source_report": args.retrieval_report,
        "mode": args.mode,
        "proxy_faithfulness": sum(overall) / len(overall) if overall else 0.0,
        "negation_failure_rate": sum(negation_flags) / len(negation_flags) if negation_flags else 0.0,
        "by_category_proxy_faithfulness": {k: (sum(v) / len(v) if v else 0.0) for k, v in by_cat.items()},
    }

    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Saved report to: {out}")


if __name__ == "__main__":
    main()

