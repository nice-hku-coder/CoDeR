from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

"""基于 BEIR 的 LLM 改写数据集构建脚本。

脚本会按类别改写 query，检索候选文档，再把部分文档改写为
满足/违反约束的版本，最后导出 corpus、benchmark、audit 三个 JSONL 文件。
"""

@dataclass(frozen=True)
class DatasetPlan:
    """数据集与约束类别的映射配置。"""

    category: str
    beir_name: str


DATASET_PLANS: tuple[DatasetPlan, ...] = (
    DatasetPlan(category="numeric", beir_name="fiqa"),
    DatasetPlan(category="exclusion", beir_name="dbpedia-entity"),
    DatasetPlan(category="negation", beir_name="scifact"),
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数：路径、检索配置、LLM 配置与输出路径。"""

    script_path = Path(__file__).resolve()
    workspace_root = script_path.parents[2]
    coder_root = script_path.parents[1]

    parser = argparse.ArgumentParser(
        description=(
            "Build BEIR-based LLM rewrite dataset: rewrite query by category, "
            "retrieve top-12 docs, rewrite 2 docs to satisfy and 1 doc to violate constraint."
        )
    )

    parser.add_argument(
        "--beir-root",
        type=str,
        default=str(workspace_root / "beir"),
        help="Root folder that contains fiqa/dbpedia-entity/scifact.",
    )
    parser.add_argument(
        "--index-root",
        type=str,
        required=True,
        help=(
            "pyserini Lucene 索引根目录，例如 <index_root>/fiqa、<index_root>/dbpedia-entity、<index_root>/scifact。"
        ),
    )
    parser.add_argument(
        "--qrels-split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Preferred qrels split.",
    )

    parser.add_argument(
        "--queries-per-dataset",
        type=int,
        default=200,
        help="Target number of successful samples for each dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--min-query-words",
        type=int,
        default=4,
        help="Minimum query length in words.",
    )
    parser.add_argument(
        "--max-query-words",
        type=int,
        default=20,
        help="Maximum query length in words.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Candidate pool size per query after qrels-priority + BM25 fill.",
    )
    parser.add_argument(
        "--final-docs",
        type=int,
        default=5,
        help="Final topical docs per query.",
    )
    parser.add_argument(
        "--qrels-priority-count",
        type=int,
        default=2,
        help="How many qrels docs to prioritize in the final five docs.",
    )
    parser.add_argument(
        "--duplicate-jaccard-threshold",
        type=float,
        default=0.88,
        help="Jaccard threshold for near-duplicate filtering while selecting docs.",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api.vectorengine.ai/v1",
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="API key. If empty, read env VECTORENGINE_API_KEY.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4",
        help="Model used for query rewrite and doc rewrite.",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=3,
        help="Max retries for each LLM call.",
    )

    parser.add_argument(
        "--output-corpus",
        type=str,
        default=str(coder_root / "data" / "processed" / "retrieval_corpus_beir_llm_rewrite.jsonl"),
        help="Output retrieval corpus JSONL.",
    )
    parser.add_argument(
        "--output-benchmark",
        type=str,
        default=str(coder_root / "data" / "processed" / "retrieval_benchmark_beir_llm_rewrite.jsonl"),
        help="Output retrieval benchmark JSONL.",
    )
    parser.add_argument(
        "--output-audit",
        type=str,
        default=str(coder_root / "data" / "processed" / "retrieval_benchmark_beir_llm_rewrite_audit.jsonl"),
        help="Output audit JSONL.",
    )

    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件为字典列表。"""

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """写出 JSONL 文件，必要时自动创建父目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_space(text: str) -> str:
    """把文本归一化为单行、单空格格式。"""

    return " ".join(text.replace("\n", " ").split()).strip()


def tokenize_for_set(text: str) -> set[str]:
    """用于去重的轻量分词（返回 token 集合）。"""

    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    """计算两个 token 集合的 Jaccard 相似度。"""

    if not a and not b:
        return 1.0
    denom = len(a | b)
    if denom == 0:
        return 0.0
    return len(a & b) / denom


def is_medium_query(text: str, min_words: int, max_words: int) -> bool:
    """按词数过滤 query，保留中等长度样本。"""

    n_words = len(text.split())
    return min_words <= n_words <= max_words


def load_queries(dataset_dir: Path) -> Dict[str, str]:
    """读取 BEIR queries.jsonl，返回 {query_id: query_text}。"""

    rows = read_jsonl(dataset_dir / "queries.jsonl")
    out: Dict[str, str] = {}
    for row in rows:
        qid = str(row.get("_id", "")).strip()
        text = normalize_space(str(row.get("text", "")))
        if qid and text:
            out[qid] = text
    return out


def load_qrels(dataset_dir: Path, preferred_split: str) -> Dict[str, Dict[str, float]]:
    """读取 qrels TSV，返回 {query_id: {doc_id: score}}。

    若指定 split 不存在，则回退到目录中第一个可用 TSV。
    """

    qrels_dir = dataset_dir / "qrels"
    preferred = qrels_dir / f"{preferred_split}.tsv"

    if preferred.exists():
        qrels_file = preferred
    else:
        candidates = sorted(qrels_dir.glob("*.tsv"))
        if not candidates:
            raise FileNotFoundError(f"No qrels files found in {qrels_dir}")
        qrels_file = candidates[0]
        print(f"[WARN] split={preferred_split} missing for {dataset_dir.name}; fallback to {qrels_file.name}")

    out: Dict[str, Dict[str, float]] = {}
    with qrels_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = str(row["query-id"]).strip()
            did = str(row["corpus-id"]).strip()
            score = float(row["score"])
            out.setdefault(qid, {})[did] = score
    return out


def safe_json_extract(text: str) -> Dict[str, Any]:
    """尽力从模型返回文本中提取 JSON 对象。"""

    text = text.strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        fragment = text[start : end + 1]
        try:
            obj = json.loads(fragment)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return {}
    return {}


def parse_doc_raw(raw: str) -> Dict[str, str]:
    """解析 BEIR 文档行，统一产出 title/text。"""

    raw = (raw or "").strip()
    if not raw:
        return {"title": "", "text": ""}

    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            title = normalize_space(str(obj.get("title", "")))
            text = normalize_space(str(obj.get("text", "")))
            if text:
                return {"title": title, "text": text}

            contents = normalize_space(str(obj.get("contents", "")))
            if contents:
                return {"title": title, "text": contents}
        except json.JSONDecodeError:
            pass

    return {"title": "", "text": normalize_space(raw)}


class LuceneBM25Retriever:
    """基于 Pyserini Lucene 索引的 BM25 检索器。"""

    def __init__(self, index_path: Path) -> None:
        """初始化 Lucene 检索器。"""

        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("需要 pyserini。请先安装: pip install pyserini") from exc

        if not index_path.exists():
            raise FileNotFoundError(f"Lucene 索引不存在: {index_path}")
        self.searcher = LuceneSearcher(str(index_path))

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """执行 BM25 检索并返回 (doc_id, score)。"""

        hits = self.searcher.search(query, k=k)
        return [(str(h.docid), float(h.score)) for h in hits]

    def get_doc(self, doc_id: str) -> Dict[str, str]:
        """按 doc_id 读取并解析原始文档。"""

        doc = self.searcher.doc(doc_id)
        if doc is None:
            return {"title": "", "text": ""}
        return parse_doc_raw(doc.raw() or "")


def make_client(base_url: str, api_key: str) -> OpenAI:
    """创建 OpenAI 兼容客户端并检查 API Key。"""

    key = api_key.strip()
    if not key:
        raise ValueError("Missing API key. Please pass --api-key or set VECTORENGINE_API_KEY.")
    return OpenAI(base_url=base_url, api_key=key)


def call_chat_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
    max_retries: int,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    """调用 chat-completions，并在重试机制下确保返回 JSON 对象。"""

    user_prompt = json.dumps(user_payload, ensure_ascii=False)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
            parsed = safe_json_extract(text)
            if parsed:
                return parsed
            raise ValueError("Model response is not parseable JSON.")
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.5 * attempt)

    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")


def build_query_rewrite_prompt(category: str, query: str) -> Dict[str, Any]:
    """构造按类别改写 query 的结构化提示。"""

    return {
        "task": "Rewrite query into a constraint-aware version for the target category",
        "category": category,
        "original_query": query,
        "rules": [
            "Keep topic entities and core topical terms",
            "Inject explicit constraint expression only",
            "Do not change retrieval topic",
            "Return JSON only",
        ],
        "return_schema": {
            "rewritten_query": "string",
            "constraint_summary": "string",
        },
    }


def build_doc_rewrite_prompt(category: str, rewritten_query: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """构造文档 satisfy/violate 改写的结构化提示。"""

    return {
        "task": "Rewrite selected docs while preserving topic relevance",
        "category": category,
        "rewritten_query": rewritten_query,
        "docs": docs,
        "rules": [
            "Use full document text as input context",
            "Choose exactly 2 docs and rewrite them to fully satisfy the rewritten query constraint",
            "Choose exactly 1 different doc and rewrite it to violate the rewritten query constraint",
            "Do not change overall topical relevance",
            "Keep writing style natural and coherent",
            "Return JSON only",
        ],
        "return_schema": {
            "satisfy_doc_ids": ["doc_id_1", "doc_id_2"],
            "violate_doc_id": "doc_id_3",
            "rewritten_docs": [
                {
                    "doc_id": "string",
                    "label": "satisfy|violate",
                    "rewritten_text": "string",
                    "reason": "string",
                }
            ],
        },
    }


def validate_doc_rewrite(
    resp: Dict[str, Any],
    available_doc_ids: Sequence[str],
) -> Tuple[bool, str, List[str], str, Dict[str, Dict[str, str]]]:
    """校验文档改写返回格式，并提取改写映射。

    约束：
    - 必须恰好 2 个 satisfy 文档
    - 必须恰好 1 个 violate 文档
    - 选择的文档 id 必须都在输入文档集合内
    - rewritten_docs 必须提供对应改写文本
    """

    doc_id_set = set(available_doc_ids)

    satisfy_ids = [str(x).strip() for x in resp.get("satisfy_doc_ids", []) if str(x).strip()]
    violate_id = str(resp.get("violate_doc_id", "")).strip()

    if len(satisfy_ids) != 2:
        return False, "satisfy_doc_ids count must be exactly 2", [], "", {}

    if violate_id == "":
        return False, "violate_doc_id is empty", [], "", {}

    if len(set(satisfy_ids + [violate_id])) != 3:
        return False, "satisfy/violate doc ids must be distinct", [], "", {}

    if any(did not in doc_id_set for did in satisfy_ids + [violate_id]):
        return False, "selected doc ids contain unknown ids", [], "", {}

    rewritten_docs = resp.get("rewritten_docs", [])
    rewritten_map: Dict[str, Dict[str, str]] = {}
    for item in rewritten_docs:
        did = str(item.get("doc_id", "")).strip()
        label = str(item.get("label", "")).strip().lower()
        text = normalize_space(str(item.get("rewritten_text", "")))
        reason = normalize_space(str(item.get("reason", "")))
        if did and label in {"satisfy", "violate"} and text:
            rewritten_map[did] = {"label": label, "text": text, "reason": reason}

    required_ids = set(satisfy_ids + [violate_id])
    if not required_ids.issubset(set(rewritten_map.keys())):
        return False, "rewritten_docs missing required ids", [], "", {}

    for did in satisfy_ids:
        if rewritten_map[did]["label"] != "satisfy":
            return False, "label mismatch for satisfy doc", [], "", {}
    if rewritten_map[violate_id]["label"] != "violate":
        return False, "label mismatch for violate doc", [], "", {}

    return True, "", satisfy_ids, violate_id, rewritten_map


def select_candidate_topk(
    retriever: LuceneBM25Retriever,
    query: str,
    qrels_for_query: Dict[str, float],
    top_k: int,
) -> List[str]:
    """构造候选集：qrels 优先 + BM25 补齐到 top-k。"""

    bm25_hits = retriever.search(query, k=top_k)
    bm25_doc_ids = [did for did, _ in bm25_hits]

    qrels_ranked = [did for did, _ in sorted(qrels_for_query.items(), key=lambda kv: kv[1], reverse=True)]

    merged: List[str] = []
    seen: set[str] = set()
    for did in qrels_ranked + bm25_doc_ids:
        if did in seen:
            continue
        seen.add(did)
        merged.append(did)
        if len(merged) >= top_k:
            break

    return merged


def select_final_docs(
    candidate_doc_ids: Sequence[str],
    qrels_for_query: Dict[str, float],
    doc_texts: Dict[str, str],
    final_docs: int,
    qrels_priority_count: int,
    dup_threshold: float,
) -> List[str]:
    """选择最终文档子集：qrels 优先 + 近重复过滤。"""

    if final_docs <= 0:
        return []

    qrels_ranked = [did for did, _ in sorted(qrels_for_query.items(), key=lambda kv: kv[1], reverse=True)]
    qrels_ranked = [did for did in qrels_ranked if did in candidate_doc_ids]

    selected: List[str] = []
    selected_token_sets: List[set[str]] = []

    def can_add(did: str) -> bool:
        toks = tokenize_for_set(doc_texts.get(did, ""))
        if not toks:
            return False
        for old in selected_token_sets:
            if jaccard(toks, old) >= dup_threshold:
                return False
        return True

    def add_doc(did: str) -> bool:
        if did in selected:
            return False
        if not can_add(did):
            return False
        selected.append(did)
        selected_token_sets.append(tokenize_for_set(doc_texts.get(did, "")))
        return True

    for did in qrels_ranked:
        add_doc(did)
        if len(selected) >= qrels_priority_count:
            break

    for did in candidate_doc_ids:
        if len(selected) >= final_docs:
            break
        add_doc(did)

    if len(selected) < final_docs:
        # 最后放宽一次去重，避免该 query 因文档不足被丢弃。
        for did in candidate_doc_ids:
            if did in selected:
                continue
            text = doc_texts.get(did, "")
            if not text:
                continue
            selected.append(did)
            if len(selected) >= final_docs:
                break

    return selected[:final_docs]


def rewrite_query(
    client: OpenAI,
    model: str,
    category: str,
    source_query: str,
    max_retries: int,
) -> Tuple[str, Dict[str, Any]]:
    """把原 query 改写为目标类别的约束 query。"""

    system_prompt = "You are a data construction assistant. Return strict JSON only."
    payload = build_query_rewrite_prompt(category=category, query=source_query)
    resp = call_chat_json(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_payload=payload,
        max_retries=max_retries,
        temperature=0.2,
    )
    rewritten = normalize_space(str(resp.get("rewritten_query", "")))
    if not rewritten:
        raise RuntimeError("Empty rewritten_query from LLM")
    return rewritten, resp


def rewrite_docs(
    client: OpenAI,
    model: str,
    category: str,
    rewritten_query: str,
    docs: List[Dict[str, Any]],
    max_retries: int,
) -> Tuple[List[str], str, Dict[str, Dict[str, str]], Dict[str, Any]]:
    """将选中文档改写为 satisfy/violate 版本并做结果校验。"""

    system_prompt = "You are a data construction assistant. Return strict JSON only."
    payload = build_doc_rewrite_prompt(category=category, rewritten_query=rewritten_query, docs=docs)

    last_err = ""
    for _ in range(max_retries):
        resp = call_chat_json(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_payload=payload,
            max_retries=max_retries,
            temperature=0.2,
        )
        ok, err_msg, satisfy_ids, violate_id, rewritten_map = validate_doc_rewrite(
            resp=resp,
            available_doc_ids=[d["doc_id"] for d in docs],
        )
        if ok:
            return satisfy_ids, violate_id, rewritten_map, resp
        last_err = err_msg
        time.sleep(0.5)

    raise RuntimeError(f"Invalid doc rewrite response: {last_err}")


def main() -> None:
    """执行端到端数据构建流程。

    流程概览：
    1) 读取 queries/qrels/corpus 并过滤候选 query
    2) 按类别用 LLM 改写 query
    3) 用同款 BM25 检索候选并选出最终文档
    4) 用 LLM 生成 satisfy/violate 文档改写
    5) 组装 corpus/benchmark 并记录 audit
    6) 导出 JSONL 文件
    """

    args = parse_args()
    rng = random.Random(args.seed)

    api_key = args.api_key.strip() or ""
    if not api_key:
        api_key = str(__import__("os").environ.get("VECTORENGINE_API_KEY", "")).strip()

    client = make_client(base_url=args.base_url, api_key=api_key)

    beir_root = Path(args.beir_root)
    index_root = Path(args.index_root)

    corpus_rows: List[Dict[str, Any]] = []
    benchmark_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    # 对输出语料按文本去重，避免重复文档占用 doc_id。
    text_key_to_doc_id: Dict[Tuple[str, str], str] = {}
    doc_counter = 0
    query_counter = 0

    def upsert_corpus_doc(
        *,
        title: str,
        text: str,
        source_dataset: str,
        source_doc_id: str,
        source_query_id: str,
        rewritten: bool,
        rewrite_label: str,
    ) -> str:
        """插入去重后的语料文档并返回输出 doc_id。"""

        nonlocal doc_counter
        key = (normalize_space(title), normalize_space(text))
        if key in text_key_to_doc_id:
            return text_key_to_doc_id[key]

        doc_counter += 1
        doc_id = f"doc-{doc_counter:07d}"
        text_key_to_doc_id[key] = doc_id
        corpus_rows.append(
            {
                "doc_id": doc_id,
                "title": normalize_space(title),
                "text": normalize_space(text),
                "source_dataset": source_dataset,
                "source_doc_id": source_doc_id,
                "source_query_id": source_query_id,
                "rewritten": rewritten,
                "rewrite_label": rewrite_label,
            }
        )
        return doc_id

    for plan in DATASET_PLANS:
        dataset_dir = beir_root / plan.beir_name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

        retriever = LuceneBM25Retriever(index_root / plan.beir_name)
        queries = load_queries(dataset_dir)
        qrels = load_qrels(dataset_dir, preferred_split=args.qrels_split)

        candidate_qids = [
            qid
            for qid in qrels.keys()
            if qid in queries
            and qrels[qid]
            and is_medium_query(queries[qid], args.min_query_words, args.max_query_words)
        ]
        rng.shuffle(candidate_qids)

        print(
            f"[INFO] dataset={plan.beir_name} category={plan.category} "
            f"candidate_queries={len(candidate_qids)} target={args.queries_per_dataset}"
        )

        success_count = 0
        for qid in candidate_qids:
            if success_count >= args.queries_per_dataset:
                break

            source_query = queries[qid]
            qrels_for_query = qrels[qid]

            try:
                rewritten_query, query_rewrite_raw = rewrite_query(
                    client=client,
                    model=args.model,
                    category=plan.category,
                    source_query=source_query,
                    max_retries=args.llm_max_retries,
                )

                candidate_doc_ids = select_candidate_topk(
                    retriever=retriever,
                    query=rewritten_query,
                    qrels_for_query=qrels_for_query,
                    top_k=args.top_k,
                )
                if len(candidate_doc_ids) < args.final_docs:
                    continue

                doc_cache: Dict[str, Dict[str, str]] = {}
                for did in candidate_doc_ids:
                    doc_obj = retriever.get_doc(did)
                    title = normalize_space(doc_obj.get("title", ""))
                    text = normalize_space(doc_obj.get("text", ""))
                    if not text:
                        continue
                    doc_cache[did] = {"title": title, "text": text}

                candidate_doc_ids = [did for did in candidate_doc_ids if did in doc_cache]
                if len(candidate_doc_ids) < args.final_docs:
                    continue

                final_doc_ids = select_final_docs(
                    candidate_doc_ids=candidate_doc_ids,
                    qrels_for_query=qrels_for_query,
                    doc_texts={did: doc_cache[did]["text"] for did in candidate_doc_ids},
                    final_docs=args.final_docs,
                    qrels_priority_count=args.qrels_priority_count,
                    dup_threshold=args.duplicate_jaccard_threshold,
                )
                if len(final_doc_ids) < args.final_docs:
                    continue

                llm_docs = [
                    {
                        "doc_id": did,
                        "title": doc_cache[did]["title"],
                        "text": doc_cache[did]["text"],
                    }
                    for did in final_doc_ids
                ]

                satisfy_ids, violate_id, rewritten_map, doc_rewrite_raw = rewrite_docs(
                    client=client,
                    model=args.model,
                    category=plan.category,
                    rewritten_query=rewritten_query,
                    docs=llm_docs,
                    max_retries=args.llm_max_retries,
                )

                topical_doc_ids: List[str] = []
                constraint_doc_ids: List[str] = []
                graded_relevance: Dict[str, int] = {}
                doc_debug_rows: List[Dict[str, Any]] = []

                for did in final_doc_ids:
                    src_title = doc_cache[did]["title"]
                    src_text = doc_cache[did]["text"]

                    if did in rewritten_map:
                        new_text = rewritten_map[did]["text"]
                        rewrite_label = rewritten_map[did]["label"]
                        rewritten = True
                    else:
                        new_text = src_text
                        rewrite_label = "topical"
                        rewritten = False

                    out_doc_id = upsert_corpus_doc(
                        title=src_title,
                        text=new_text,
                        source_dataset=plan.beir_name,
                        source_doc_id=did,
                        source_query_id=qid,
                        rewritten=rewritten,
                        rewrite_label=rewrite_label,
                    )

                    topical_doc_ids.append(out_doc_id)
                    if did in satisfy_ids:
                        graded_relevance[out_doc_id] = 2
                        constraint_doc_ids.append(out_doc_id)
                    elif did == violate_id:
                        graded_relevance[out_doc_id] = 0
                    else:
                        graded_relevance[out_doc_id] = 1

                    doc_debug_rows.append(
                        {
                            "source_doc_id": did,
                            "out_doc_id": out_doc_id,
                            "rewritten": rewritten,
                            "rewrite_label": rewrite_label,
                            "reason": rewritten_map.get(did, {}).get("reason", ""),
                        }
                    )

                if len(constraint_doc_ids) != 2:
                    continue
                if len(topical_doc_ids) != args.final_docs:
                    continue

                query_counter += 1
                out_query_id = f"q-{query_counter:07d}"

                benchmark_rows.append(
                    {
                        "query_id": out_query_id,
                        "query": rewritten_query,
                        "original_query": source_query,
                        "category": plan.category,
                        "source_dataset": plan.beir_name,
                        "source_query_id": qid,
                        "topical_relevant_doc_ids": topical_doc_ids,
                        "constraint_satisfying_doc_ids": constraint_doc_ids,
                        "graded_relevance": graded_relevance,
                    }
                )

                audit_rows.append(
                    {
                        "query_id": out_query_id,
                        "source_dataset": plan.beir_name,
                        "source_query_id": qid,
                        "category": plan.category,
                        "original_query": source_query,
                        "rewritten_query": rewritten_query,
                        "query_rewrite_raw": query_rewrite_raw,
                        "doc_rewrite_raw": doc_rewrite_raw,
                        "doc_mappings": doc_debug_rows,
                        "satisfy_source_doc_ids": satisfy_ids,
                        "violate_source_doc_id": violate_id,
                    }
                )

                success_count += 1
                if success_count % 10 == 0:
                    print(
                        f"[PROGRESS] dataset={plan.beir_name} success={success_count}/{args.queries_per_dataset}"
                    )

            except Exception as exc:
                audit_rows.append(
                    {
                        "source_dataset": plan.beir_name,
                        "source_query_id": qid,
                        "category": plan.category,
                        "original_query": source_query,
                        "error": str(exc),
                    }
                )
                continue

        print(
            f"[DONE] dataset={plan.beir_name} built={success_count}/{args.queries_per_dataset}"
        )

    write_jsonl(Path(args.output_corpus), corpus_rows)
    write_jsonl(Path(args.output_benchmark), benchmark_rows)
    write_jsonl(Path(args.output_audit), audit_rows)

    print(f"Saved corpus: {len(corpus_rows)} -> {args.output_corpus}")
    print(f"Saved benchmark: {len(benchmark_rows)} -> {args.output_benchmark}")
    print(f"Saved audit: {len(audit_rows)} -> {args.output_audit}")


if __name__ == "__main__":
    main()
