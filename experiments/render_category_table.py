from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def to_percent(x: float) -> str:
    return f"{x * 100:.2f}%"


def render_single(title: str, report: dict) -> str:
    lines = [f"## {title}", ""]
    lines.append(
        f"- Queries: {report['num_queries']}, Top-k: {report['top_k']}, "
        f"alpha={report['alpha']}, tau={report['tau']}"
    )
    lines.append(
        f"- Overall: Vanilla {to_percent(report['overall']['vanilla_ccr'])} -> "
        f"Dual {to_percent(report['overall']['dual_ccr'])} "
        f"(+{to_percent(report['overall']['improvement'])})"
    )
    lines.append("")
    lines.append("| Category | Count | Vanilla CCR | Dual CCR | Improvement |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in report["by_category"]:
        lines.append(
            f"| {row['category']} | {row['count']} | {to_percent(row['vanilla_ccr'])} | "
            f"{to_percent(row['dual_ccr'])} | +{to_percent(row['improvement'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render markdown tables from category report JSON files.")
    parser.add_argument("--report-a", type=str, required=True)
    parser.add_argument("--title-a", type=str, default="Setting A")
    parser.add_argument("--report-b", type=str, default="")
    parser.add_argument("--title-b", type=str, default="Setting B")
    parser.add_argument("--output-md", type=str, required=True)
    args = parser.parse_args()

    md_parts = []
    report_a = load_report(Path(args.report_a))
    md_parts.append(render_single(args.title_a, report_a))

    if args.report_b:
        report_b = load_report(Path(args.report_b))
        md_parts.append(render_single(args.title_b, report_b))

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md_parts), encoding="utf-8")
    print(f"Saved markdown table to: {out}")


if __name__ == "__main__":
    main()
