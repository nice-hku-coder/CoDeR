from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOTIVATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MOTIVATION_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import (
    FIGURES_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    ensure_project_dirs,
    load_sentence_encoder,
    read_jsonl,
    resolve_local_model_path,
)
from experiments.metrics import ndcg_at_k, recall_at_k
from utils import doc2fol, query2fol, updated_embeddings
from motivation.rankers import build_bm25_ranker, build_dense_ranker, build_hyde_ranker

DEFAULT_MODELS = {
    "TopicEncoder": str(PROJECT_ROOT / "models" / "sentence-transformers_all-MiniLM-L6-v2"),
    "BGE": str(PROJECT_ROOT / "models" / "BAAI_bge-large-en-v1.5"),
    "Contriever": str(PROJECT_ROOT / "models" / "facebook_contriever"),
    "ConstraintEncoder": str(PROJECT_ROOT / "outputs" / "checkpoints" / "constraint-encoder-v1"),
    "NSIR_Encoder": str(PROJECT_ROOT / "models" / "BAAI_bge-large-en-v1.5"),
    "NSIR_Generator": "gpt-4o",
    "HyDE_Generator": "gpt-5",
}
NSIR_BASE_URL = "https://api.vectorengine.ai/v1"
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
    parser.add_argument("--topic-model", type=str, default=DEFAULT_MODELS["TopicEncoder"])
    parser.add_argument("--bge-model", type=str, default=DEFAULT_MODELS["BGE"])
    parser.add_argument("--contriever-model", type=str, default=DEFAULT_MODELS["Contriever"])
    parser.add_argument("--constraint-model", type=str, default=DEFAULT_MODELS["ConstraintEncoder"])
    parser.add_argument("--dual-alpha", type=float, default=0.0)
    parser.add_argument("--dual-tau", type=float, default=0.6)
    parser.add_argument("--dual-retrieve-k", type=int, default=100)
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
        help="Enable HyDE baseline",
    )
    parser.add_argument("--hyde-api-key", type=str, default="", help="API key for OpenAI-compatible endpoint.")
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
    parser.add_argument("--enable-nsir", action="store_true", help="Enable NS-IR baseline")
    parser.add_argument("--nsir-api-key", type=str, default="", help="API key for OpenAI-compatible endpoint.")
    parser.add_argument("--nsir-base-url", type=str, default=NSIR_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--nsir-encoder-model", type=str, default=DEFAULT_MODELS["NSIR_Encoder"])
    parser.add_argument("--nsir-generator-model", type=str, default=DEFAULT_MODELS["NSIR_Generator"])
    parser.add_argument("--nsir-retrieve-k", type=int, default=100)
    parser.add_argument("--nsir-max-tokens", type=int, default=512)
    parser.add_argument("--nsir-temperature", type=float, default=0.0)
    parser.add_argument("--nsir-top-p", type=float, default=1.0)
    parser.add_argument("--nsir-distortion", type=float, default=0.2)
    parser.add_argument("--nsir-sinkhorn", action="store_true")
    parser.add_argument("--nsir-epsilon", type=float, default=0.1)
    parser.add_argument("--nsir-stop-thr", type=float, default=1e-6)
    parser.add_argument("--nsir-num-iter-max", type=int, default=1000)
    parser.add_argument(
        "--nsir-cache-file",
        type=str,
        default=str(REPORTS_DIR / "motivation" / "nsir_generations.jsonl"),
    )
    parser.add_argument(
        "--nsir-wait-till-success",
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


def _cosine_score(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=1).item() + 1.0) / 2.0)


def build_dual_ranker(
    topic_model_name: str,
    constraint_model_name: str,
    corpus: list[dict],
    alpha: float,
    tau: float,
    retrieve_k: int,
) -> Callable[[str], list[int]]:
    topic_model = load_sentence_encoder(topic_model_name)
    constraint_model = load_sentence_encoder(constraint_model_name)
    doc_texts = [row["text"] for row in corpus]
    topic_doc_emb = topic_model.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    constraint_doc_emb = constraint_model.encode(
        doc_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    )

    def rank(query: str) -> list[int]:
        q_topic = topic_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        q_constraint = constraint_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        topic_scores = np.dot(topic_doc_emb, q_topic)
        constraint_scores = np.dot(constraint_doc_emb, q_constraint)

        topic_idx = np.argsort(-topic_scores)
        retrieve_idx = topic_idx[: min(retrieve_k, len(topic_idx))]
        if len(retrieve_idx) == 0:
            return []

        final_scores = alpha * topic_scores[retrieve_idx] + (1.0 - alpha) * constraint_scores[retrieve_idx]
        keep_local = [j for j, i in enumerate(retrieve_idx) if float(constraint_scores[i]) >= tau]
        if keep_local:
            kept_idx = retrieve_idx[keep_local]
            kept_scores = final_scores[keep_local]
            order = np.argsort(-kept_scores)
            fused_idx = kept_idx[order]
        else:
            fused_idx = retrieve_idx

        fused_set = set(fused_idx)
        return list(fused_idx) + [i for i in topic_idx if i not in fused_set]

    return rank


def build_nsir_ranker(
    corpus: list[dict],
    encoder_model_name: str,
    generator_model_name: str,
    api_key: str,
    base_url: str,
    device: str,
    retrieve_k: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    distortion_ratio: float,
    sinkhorn: bool,
    epsilon: float,
    stop_thr: float,
    num_itermax: int,
    wait_till_success: bool,
) -> Callable[[str], list[int]]:
    resolved_encoder_model = resolve_local_model_path(encoder_model_name)
    Settings.embed_model = HuggingFaceEmbedding(model_name=resolved_encoder_model, device=device)

    doc_texts = []
    corpus_dict: dict[str, int] = {}
    for idx, row in enumerate(corpus):
        text = row["text"]
        doc_texts.append(text)
        corpus_dict[text] = idx

    documents = [Document(text=text) for text in doc_texts]
    splitter = SentenceSplitter(chunk_size=1000000)
    index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
    VIretriever = VectorIndexRetriever(index=index, similarity_top_k=retrieve_k)

    tokenizer = AutoTokenizer.from_pretrained(resolved_encoder_model)
    model = AutoModel.from_pretrained(resolved_encoder_model).to(device)
    model.eval()
    args_namespace = SimpleNamespace(api_key=api_key, base_url=base_url)

    def rank(query: str) -> list[int]:
        bge_nodes = VIretriever.retrieve(query)
        if len(bge_nodes) == 0:
            return []

        reranked: list[tuple[int, float]] = []
        doc_reprs: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        doc_iterator = tqdm(bge_nodes, desc="NS-IR docs", leave=False, total=len(bge_nodes))
        for node in doc_iterator:
            doc_text = node.text
            doc_id = corpus_dict[doc_text]
            doc_premise = doc2fol(
                doc_text,
                args_namespace,
            )
            doc_embedding, doc_word_embedding = updated_embeddings(
                model,
                tokenizer,
                doc_text,
                doc_premise,
                device=device,
                distortion_ratio=distortion_ratio,
                sinkhorn=sinkhorn,
                epsilon=epsilon,
                stop_thr=stop_thr,
                num_itermax=num_itermax,
            )
            doc_reprs.append((doc_id, doc_embedding, doc_word_embedding))

        query_premise = query2fol(
            query,
            args_namespace,
        )
        query_embedding, query_word_embedding = updated_embeddings(
            model,
            tokenizer,
            query,
            query_premise,
            device=device,
            distortion_ratio=distortion_ratio,
            sinkhorn=sinkhorn,
            epsilon=epsilon,
            stop_thr=stop_thr,
            num_itermax=num_itermax,
        )

        for idx, doc_embedding, doc_word_embedding in doc_reprs:
            score = _cosine_score(query_embedding, doc_embedding) + _cosine_score(query_word_embedding, doc_word_embedding)
            reranked.append((idx, score))

        reranked.sort(key=lambda item: item[1], reverse=True)
        fused_idx = [idx for idx, _ in reranked]
        return fused_idx

    return rank


def compute_method_report(method_name: str, benchmark: list[dict], doc_ids: list[str], rank_fn, max_k: int) -> dict:
    overall_vr: dict[int, list[float]] = {k: [] for k in K_VALUES if k <= max_k}
    category_vr5: dict[str, list[float]] = {cat: [] for cat in TARGET_CATEGORIES}
    recall5_values: list[float] = []
    ndcg5_values: list[float] = []
    fvr_values: list[int] = []
    per_query: list[dict] = []

    iterator = tqdm(benchmark, desc=f"Evaluating {method_name}", total=len(benchmark), leave=False) if method_name == "NS-IR" else benchmark

    for item in iterator:
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
        "EncoderA": {"color": "#4c78a8", "linestyle": "-", "marker": "s"},
        "EncoderB": {"color": "#9467bd", "linestyle": "-", "marker": "P"},
        "DualFusion": {"color": "#d62728", "linestyle": "-", "marker": "X"},
        "NS-IR": {"color": "#8c564b", "linestyle": "-", "marker": "v"},
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
        "EncoderA": "#aec7e8",
        "EncoderB": "#c5b0d5",
        "DualFusion": "#ff9896",
        "NS-IR": "#c49c94",
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
    width = min(0.18, 0.75 / max(1, len(methods)))
    colors = {
        "BM25": "#7f7f7f",
        "BGE": "#1f77b4",
        "Contriever": "#ff7f0e",
        "EncoderA": "#4c78a8",
        "EncoderB": "#9467bd",
        "DualFusion": "#d62728",
        "NS-IR": "#8c564b",
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
    if args.enable_nsir and not args.nsir_api_key:
        raise ValueError("--nsir-api-key is required when --enable-nsir is set.")

    fig_dir = Path(args.fig_dir)
    report_path = Path(args.report_file)
    doc_ids = [row["doc_id"] for row in corpus]

    rankers = {
        "EncoderA": build_dense_ranker(args.topic_model, corpus),
        "EncoderB": build_dense_ranker(args.constraint_model, corpus),
        "DualFusion": build_dual_ranker(
            topic_model_name=args.topic_model,
            constraint_model_name=args.constraint_model,
            corpus=corpus,
            alpha=args.dual_alpha,
            tau=args.dual_tau,
            retrieve_k=args.dual_retrieve_k,
        ),
        "BM25": build_bm25_ranker(corpus),
        "BGE": build_dense_ranker(args.bge_model, corpus, query_prefix="Represent this sentence for searching relevant passages: "),
        "Contriever": build_dense_ranker(args.contriever_model, corpus),
    }
    if args.enable_nsir:
        nsir_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        rankers["NS-IR"] = build_nsir_ranker(
            corpus=corpus,
            encoder_model_name=args.nsir_encoder_model,
            generator_model_name=args.nsir_generator_model,
            api_key=args.nsir_api_key,
            base_url=args.nsir_base_url,
            device=nsir_device,
            retrieve_k=args.nsir_retrieve_k,
            max_tokens=args.nsir_max_tokens,
            temperature=args.nsir_temperature,
            top_p=args.nsir_top_p,
            distortion_ratio=args.nsir_distortion,
            sinkhorn=args.nsir_sinkhorn,
            epsilon=args.nsir_epsilon,
            stop_thr=args.nsir_stop_thr,
            num_itermax=args.nsir_num_iter_max,
            wait_till_success=args.nsir_wait_till_success,
        )
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
        "EncoderA": args.topic_model,
        "EncoderB": args.constraint_model,
        "DualFusion": {
            "topic_model": args.topic_model,
            "constraint_model": args.constraint_model,
            "alpha": args.dual_alpha,
            "tau": args.dual_tau,
            "retrieve_k": args.dual_retrieve_k,
        },
        "BM25": "BM25",
        "BGE": args.bge_model,
        "Contriever": args.contriever_model,
    }
    if args.enable_nsir:
        models["NS-IR"] = {
            "encoder": args.nsir_encoder_model,
            "generator": args.nsir_generator_model,
            "api_base_url": args.nsir_base_url or NSIR_BASE_URL,
            "retrieve_k": args.nsir_retrieve_k,
            "max_tokens": args.nsir_max_tokens,
            "temperature": args.nsir_temperature,
            "top_p": args.nsir_top_p,
            "distortion": args.nsir_distortion,
            "sinkhorn": args.nsir_sinkhorn,
            "epsilon": args.nsir_epsilon,
            "stop_thr": args.nsir_stop_thr,
            "num_itermax": args.nsir_num_iter_max,
            "implementation_note": (
                "Aligned with NS-IR-main: initial dense retrieval on the topic encoder, then query/doc FOL generation, "
                "OT-based logic alignment, connective masking, and candidate-only reranking. The local motivation utils "
                "module mirrors the original control flow and prompt parsing."
            ),
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
                "AutoQueryEncoder(pooling='mean') + mean embedding of [query] and generated hypothetical documents. "
                "Retrieval remains over the local motivation corpus "
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
            "dual_fusion": "DualFusion combines encoder A and encoder B scores with alpha weighting and tau filtering, mirroring eval_retrieval_metrics.py.",
        },
        "dual_config": {
            "topic_model": args.topic_model,
            "constraint_model": args.constraint_model,
            "alpha": args.dual_alpha,
            "tau": args.dual_tau,
            "retrieve_k": args.dual_retrieve_k,
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
