from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_project_dirs, read_jsonl
from experiments.metrics import ndcg_at_k, recall_at_k
from motivation.rankers import build_bm25_ranker, build_dense_ranker, build_hyde_ranker

DEFAULT_MODELS = {
    "BGE": "BAAI/bge-large-en-v1.5",
    "Contriever": "facebook/contriever",
    "HyDE_Generator": "gpt-5",
}
TARGET_CATEGORIES = ("negation", "exclusion", "numeric")
K_VALUES = (1, 3, 5, 10)
PRIMARY_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Motivation analysis: characterize constraint-violating evidence exposure in standard retrieval and HyDE."
    )
    parser.add_argument(
        "--benchmark-file",
        type=str,
        default=str(PROCESSED_DIR / "retrieval_benchmark_v1.jsonl"),
    )
    parser.add_argument(
        "--corpus-file",
        type=str,
        default=str(PROCESSED_DIR / "retrieval_corpus_v1.jsonl"),
    )
    parser.add_argument("--bge-model", type=str, default=DEFAULT_MODELS["BGE"])
    parser.add_argument("--contriever-model", type=str, default=DEFAULT_MODELS["Contriever"])
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument(
        "--report-file",
        type=str,
        default=str(REPORTS_DIR / "motivation" / "retrieval_failure_summary.json"),
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default=str(FIGURES_DIR / "motivation"),
    )
    parser.add_argument(
        "--enable-hyde",
        action="store_true",
        help="Enable HyDE baseline aligned with texttron/hyde while keeping the local motivation corpus.",
    )
    parser.add_argument("--hyde-api-key", type=str, default="", help="API key for GPT-5 OpenAI-compatible endpoint.")
    parser.add_argument("--hyde-encoder-model", type=str, default=DEFAULT_MODELS["Contriever"])
    parser.add_argument("--hyde-generator-model", type=str, default=DEFAULT_MODELS["HyDE_Generator"])
    parser.add_argument("--hyde-task", type=str, default="web search")
    parser.add_argument("--hyde-n", type=int, default=8, help="Number of hypothetical documents, matching official HyDE default.")
    parser.add_argument("--hyde-max-tokens", type=int, default=512)
    parser.add_argument("--hyde-temperature", type=float, default=0.7)
    parser.add_argument("--hyde-top-p", type=float, default=1.0)
    parser.add_argument(
        "--hyde-cache-file",
        type=str,
        default=str(REPORTS_DIR / "motivation" / "hyde_generations.jsonl"),
    )
    parser.add_argument(
        "--hyde-wait-till-success",
        action="store_true",
        help="Retry generation requests until success, mirroring the official implementation option.",
    )
    return parser.parse_args()


def violating_doc_set(item: dict) -> set[str]:
    topical = set(item["topical_relevant_doc_ids"])
    satisfying = set(item["constraint_satisfying_doc_ids"])
    return topical - satisfying


def violation_rate_at_k(ranked_doc_ids: list[str], violating_ids: set[str], k: int) -> float:
    topk = ranked_doc_ids[:k]
    if not topk:
        return 0.0
    return float(sum(1 for doc_id in topk if doc_id in violating_ids) / k)


def first_violating_rank(ranked_doc_ids: list[str], violating_ids: set[str], max_k: int) -> int:
    for rank, doc_id in enumerate(ranked_doc_ids[:max_k], start=1):
        if doc_id in violating_ids:
            return rank
    return max_k + 1


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def compute_method_report(method_name: str, benchmark: list[dict], doc_ids: list[str], rank_fn, max_k: int) -> dict:
    overall_vr: dict[int, list[float]] = {k: [] for k in K_VALUES if k <= max_k}
    category_vr5: dict[str, list[float]] = {cat: [] for cat in TARGET_CATEGORIES}
    recall5_values: list[float] = []
    ndcg5_values: list[float] = []
    fvr_values: list[int] = []
    per_query: list[dict] = []

    for item in benchmark:
        ranked_idx = rank_fn(item["query"])
        ranked_doc_ids = [doc_ids[i] for i in ranked_idx]
        violating_ids = violating_doc_set(item)
        category = item.get("category", "unknown")
        topical_ids = item["topical_relevant_doc_ids"]
        graded_relevance = item.get("graded_relevance", {})

        vr_row = {f"vr@{k}": violation_rate_at_k(ranked_doc_ids, violating_ids, k) for k in overall_vr}
        recall5 = recall_at_k(ranked_doc_ids, topical_ids, PRIMARY_K)
        ndcg5 = ndcg_at_k(ranked_doc_ids, graded_relevance, PRIMARY_K)
        fvr = first_violating_rank(ranked_doc_ids, violating_ids, max_k=max_k)

        recall5_values.append(recall5)
        ndcg5_values.append(ndcg5)
        fvr_values.append(fvr)

        if PRIMARY_K <= max_k and category in category_vr5:
            category_vr5[category].append(vr_row[f"vr@{PRIMARY_K}"])

        per_query.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "category": category,
                "recall@5": recall5,
                "ndcg@5": ndcg5,
                **vr_row,
                "first_violating_rank": fvr,
                "top10_doc_ids": ranked_doc_ids[:max_k],
                "violating_doc_ids": sorted(violating_ids),
            }
        )

        for k in overall_vr:
            overall_vr[k].append(vr_row[f"vr@{k}"])

    return {
        "method": method_name,
        "overall": {
            "recall@5": mean(recall5_values),
            "ndcg@5": mean(ndcg5_values),
            **{f"vr@{k}": mean(values) for k, values in overall_vr.items()},
        },
        "first_violating_rank": {
            "max_k": max_k,
            "no_violation_value": max_k + 1,
            "mean": mean(fvr_values),
            "median": median(fvr_values),
            "values": fvr_values,
        },
        "by_category": {
            category: {f"vr@{PRIMARY_K}": mean(values), "count": len(values)}
            for category, values in category_vr5.items()
        },
        "per_query": per_query,
    }


def plot_violation_rate_curves(summary: dict, out_path: Path) -> None:
    ks = sorted(
        int(metric.split("@", 1)[1])
        for metric in next(iter(summary.values()))["overall"]
        if metric.startswith("vr@")
    )
    styles = {
        "BM25": {"color": "#333333", "linestyle": "--", "marker": "o"},
        "BGE": {"color": "#1f77b4", "linestyle": "-", "marker": "s"},
        "Contriever": {"color": "#ff7f0e", "linestyle": "-", "marker": "^"},
        "HyDE": {"color": "#2ca02c", "linestyle": "-", "marker": "D"},
    }

    plt.figure(figsize=(7.4, 4.8))
    for method_name, report in summary.items():
        y = [report["overall"][f"vr@{k}"] for k in ks]
        style = styles.get(method_name, {})
        plt.plot(ks, y, label=method_name, linewidth=2, markersize=7, **style)

    plt.xlabel("k")
    plt.ylabel("Violation Rate@k")
    plt.title("Constraint-Violating Evidence Frequently Appears in Top-k")
    plt.xticks(ks)
    plt.ylim(bottom=0.0)
    plt.grid(alpha=0.25, linestyle=":")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_first_violating_rank_boxplot(summary: dict, out_path: Path) -> None:
    labels = list(summary.keys())
    values = [summary[label]["first_violating_rank"]["values"] for label in labels]
    palette = {
        "BM25": "#bbbbbb",
        "BGE": "#9ecae1",
        "Contriever": "#fdae6b",
        "HyDE": "#98df8a",
    }

    plt.figure(figsize=(7.2, 4.8))
    box = plt.boxplot(values, patch_artist=True, tick_labels=labels, showfliers=False)
    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(palette.get(label, "#cccccc"))
        patch.set_alpha(0.9)
    for median_line in box["medians"]:
        median_line.set_color("#d62728")
        median_line.set_linewidth(2)

    max_k = next(iter(summary.values()))["first_violating_rank"]["max_k"]
    plt.axhline(max_k + 1, color="#666666", linestyle=":", linewidth=1)
    plt.ylabel("First Violating Rank")
    plt.title("Constraint-Violating Evidence Often Appears Early")
    plt.ylim(0.8, max_k + 1.5)
    plt.grid(axis="y", alpha=0.25, linestyle=":")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_category_violation_bars(summary: dict, out_path: Path) -> None:
    methods = list(summary.keys())
    categories = list(TARGET_CATEGORIES)
    x = np.arange(len(categories))
    width = 0.18
    colors = {
        "BM25": "#7f7f7f",
        "BGE": "#1f77b4",
        "Contriever": "#ff7f0e",
        "HyDE": "#2ca02c",
    }

    plt.figure(figsize=(8.0, 4.8))
    for idx, method_name in enumerate(methods):
        heights = [summary[method_name]["by_category"][cat][f"vr@{PRIMARY_K}"] for cat in categories]
        offset = (idx - (len(methods) - 1) / 2) * width
        plt.bar(x + offset, heights, width=width, label=method_name, color=colors.get(method_name))

    plt.xticks(x, [cat.capitalize() for cat in categories])
    plt.ylabel("Violation Rate@5")
    plt.title("Violation Exposure Persists Across Constraint Types")
    plt.ylim(bottom=0.0)
    plt.grid(axis="y", alpha=0.25, linestyle=":")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_project_dirs()

    benchmark = read_jsonl(Path(args.benchmark_file))
    corpus = read_jsonl(Path(args.corpus_file))
    if not benchmark:
        raise RuntimeError(f"Empty benchmark: {args.benchmark_file}")
    if not corpus:
        raise RuntimeError(f"Empty corpus: {args.corpus_file}")
    if args.max_k < PRIMARY_K:
        raise ValueError(f"--max-k must be at least {PRIMARY_K} to draw the category-level Violation Rate@5 figure.")
    if args.enable_hyde and not args.hyde_api_key:
        raise ValueError("--hyde-api-key is required when --enable-hyde is set.")

    fig_dir = Path(args.fig_dir)
    report_path = Path(args.report_file)
    doc_ids = [row["doc_id"] for row in corpus]

    rankers = {
        "BM25": build_bm25_ranker(corpus),
        "BGE": build_dense_ranker(args.bge_model, corpus, query_prefix="Represent this sentence for searching relevant passages: "),
        "Contriever": build_dense_ranker(args.contriever_model, corpus),
    }
    if args.enable_hyde:
        rankers["HyDE"] = build_hyde_ranker(
            corpus=corpus,
            encoder_name=args.hyde_encoder_model,
            api_key=args.hyde_api_key,
            generator_model_name=args.hyde_generator_model,
            cache_path=args.hyde_cache_file,
            hyde_task=args.hyde_task,
            hyde_n=args.hyde_n,
            hyde_max_tokens=args.hyde_max_tokens,
            hyde_temperature=args.hyde_temperature,
            hyde_top_p=args.hyde_top_p,
            wait_till_success=args.hyde_wait_till_success,
        )

    summary = {
        method_name: compute_method_report(method_name, benchmark, doc_ids, rank_fn, max_k=args.max_k)
        for method_name, rank_fn in rankers.items()
    }

    figures = {
        "violation_rate_curve": fig_dir / "violation_rate_at_k.png",
        "first_violating_rank_boxplot": fig_dir / "first_violating_rank_boxplot.png",
        "category_violation_rate_bar": fig_dir / "violation_rate_at_5_by_category.png",
    }

    plot_violation_rate_curves(summary, figures["violation_rate_curve"])
    plot_first_violating_rank_boxplot(summary, figures["first_violating_rank_boxplot"])
    plot_category_violation_bars(summary, figures["category_violation_rate_bar"])

    models = {
        "BM25": "BM25",
        "BGE": args.bge_model,
        "Contriever": args.contriever_model,
    }
    if args.enable_hyde:
        models["HyDE"] = {
            "encoder": args.hyde_encoder_model,
            "generator": args.hyde_generator_model,
            "api_base_url": "https://api.vectorengine.ai/v1",
            "task": args.hyde_task,
            "n": args.hyde_n,
            "max_tokens": args.hyde_max_tokens,
            "temperature": args.hyde_temperature,
            "top_p": args.hyde_top_p,
            "cache_file": args.hyde_cache_file,
            "implementation_note": (
                "Aligned with texttron/hyde on the query side: web-search prompt + "
                "AutoQueryEncoder(pooling='mean') + mean embedding of [query] and generated hypothetical documents, "
                "without explicit post-mean L2 normalization. Retrieval remains over the local motivation corpus "
                "instead of the official MS MARCO Faiss index so the topical/constraint labels stay valid."
            ),
        }

    report = {
        "benchmark_file": args.benchmark_file,
        "corpus_file": args.corpus_file,
        "models": models,
        "metric_definition": {
            "violation_doc": "A topical relevant document that is not in constraint_satisfying_doc_ids.",
            "recall@5": "Recall on topical_relevant_doc_ids at top-5.",
            "ndcg@5": "nDCG@5 computed from graded_relevance labels.",
            "violation_rate_at_k": "Fraction of top-k results that are violating documents.",
            "first_violating_rank": f"Rank position of the first violating document within top-{args.max_k}; {args.max_k + 1} means none found.",
        },
        "target_categories": list(TARGET_CATEGORIES),
        "k_values": list(k for k in K_VALUES if k <= args.max_k),
        "summary": summary,
        "figures": {name: str(path) for name, path in figures.items()},
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    compact = {
        name: {
            "recall@5": report["summary"][name]["overall"]["recall@5"],
            "ndcg@5": report["summary"][name]["overall"]["ndcg@5"],
            **{metric: value for metric, value in report["summary"][name]["overall"].items() if metric.startswith("vr@")},
        }
        for name in report["summary"]
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    print(f"Saved summary to: {report_path}")

    for name, path in figures.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
