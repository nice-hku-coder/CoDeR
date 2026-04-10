from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

from experiments.common import FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_project_dirs, load_sentence_encoder, read_jsonl

# 匹配连续的字母、数字或美元符号序列
TOKEN_RE = re.compile(r"[a-zA-Z0-9$]+")
DEFAULT_MODELS = {
    "BM25": None,
    "BGE": "BAAI/bge-large-en-v1.5",
    "Contriever": "facebook/contriever",
}
TARGET_CATEGORIES = ("negation", "exclusion", "numeric")
K_VALUES = (1, 3, 5, 10)

class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if docs else 0.0
        self.tf = [Counter(d) for d in docs]
        self.df = defaultdict(int)
        for doc in docs:
            for term in set(doc):
                self.df[term] += 1
        self.N = len(docs)

    """
    逆文档频率：如果一个词在很多文档中都出现（如“的”、“是”），
    它的判别力就低；如果只在少数文档出现，它就更有代表性。
    """
    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        if self.N == 0:
            return 0.0
        tf = self.tf[doc_idx]
        dl = self.doc_len[doc_idx]
        score = 0.0
        for term in query_tokens:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            idf = self.idf(term)
            # 文档长度归一化：较长的文档可能包含更多的查询词出现次数，但这不一定意味着它更相关。
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-12))
            score += idf * (freq * (self.k1 + 1) / (denom + 1e-12))
        return score

# 分词
def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Motivation analysis: characterize constraint-violating evidence exposure in standard retrieval."
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
    return parser.parse_args()


def build_bm25_ranker(corpus: list[dict]):
    tokenized_docs = [tokenize(row["text"]) for row in corpus]
    bm25 = BM25(tokenized_docs)

    def rank(query: str) -> list[int]:
        query_tokens = tokenize(query)
        scores = [bm25.score(query_tokens, i) for i in range(len(corpus))]
        return list(np.argsort(-np.asarray(scores, dtype=np.float32)))

    return rank


def build_dense_ranker(model_name: str, corpus: list[dict], query_prefix: str = ""):
    model = load_sentence_encoder(model_name)
    doc_texts = [row["text"] for row in corpus]
    doc_emb = model.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

    def rank(query: str) -> list[int]:
        formatted_query = f"{query_prefix}{query}" if query_prefix else query
        q_emb = model.encode([formatted_query], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = np.dot(doc_emb, q_emb)
        return list(np.argsort(-scores))

    return rank


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
    fvr_values: list[int] = []
    per_query: list[dict] = []

    for item in benchmark:
        ranked_idx = rank_fn(item["query"])
        ranked_doc_ids = [doc_ids[i] for i in ranked_idx]
        violating_ids = violating_doc_set(item)
        category = item.get("category", "unknown")

        vr_row = {f"vr@{k}": violation_rate_at_k(ranked_doc_ids, violating_ids, k) for k in overall_vr}
        fvr = first_violating_rank(ranked_doc_ids, violating_ids, max_k=max_k)
        fvr_values.append(fvr)

        if 5 <= max_k and category in category_vr5:
            category_vr5[category].append(vr_row["vr@5"])

        per_query.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "category": category,
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
        "overall": {f"vr@{k}": mean(values) for k, values in overall_vr.items()},
        "first_violating_rank": {
            "max_k": max_k,
            "no_violation_value": max_k + 1,
            "mean": mean(fvr_values),
            "median": median(fvr_values),
            "values": fvr_values,
        },
        "by_category": {
            category: {"vr@5": mean(values), "count": len(values)} for category, values in category_vr5.items()
        },
        "per_query": per_query,
    }


def plot_violation_rate_curves(summary: dict, out_path: Path) -> None:
    ks = sorted(int(metric.split("@", 1)[1]) for metric in next(iter(summary.values()))["overall"])
    styles = {
        "BM25": {"color": "#333333", "linestyle": "--", "marker": "o"},
        "BGE": {"color": "#1f77b4", "linestyle": "-", "marker": "s"},
        "Contriever": {"color": "#ff7f0e", "linestyle": "-", "marker": "^"},
    }

    plt.figure(figsize=(7.2, 4.8))
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
    colors = ["#bbbbbb", "#9ecae1", "#fdae6b"]

    plt.figure(figsize=(7.0, 4.8))
    box = plt.boxplot(values, patch_artist=True, tick_labels=labels, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
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
    width = 0.24
    colors = {"BM25": "#7f7f7f", "BGE": "#1f77b4", "Contriever": "#ff7f0e"}

    plt.figure(figsize=(7.6, 4.8))
    for idx, method_name in enumerate(methods):
        heights = [summary[method_name]["by_category"][cat]["vr@5"] for cat in categories]
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
    if args.max_k < 5:
        raise ValueError("--max-k must be at least 5 to draw the category-level Violation Rate@5 figure.")

    fig_dir = Path(args.fig_dir)
    report_path = Path(args.report_file)
    doc_ids = [row["doc_id"] for row in corpus]

    rankers = {
        "BM25": build_bm25_ranker(corpus),
        "BGE": build_dense_ranker(args.bge_model, corpus, query_prefix="Represent this sentence for searching relevant passages: "),
        "Contriever": build_dense_ranker(args.contriever_model, corpus),
    }

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

    report = {
        "benchmark_file": args.benchmark_file,
        "corpus_file": args.corpus_file,
        "models": {
            "BM25": "BM25",
            "BGE": args.bge_model,
            "Contriever": args.contriever_model,
        },
        "metric_definition": {
            "violation_doc": "A topical relevant document that is not in constraint_satisfying_doc_ids.",
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

    print(json.dumps({name: report["summary"][name]["overall"] for name in report["summary"]}, indent=2, ensure_ascii=False))
    print(f"Saved summary to: {report_path}")
    for name, path in figures.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
