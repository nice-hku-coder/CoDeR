from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from openai import OpenAI

from experiments.common import load_sentence_encoder, resolve_local_model_path

from pyserini.encode import AutoQueryEncoder

TOKEN_RE = re.compile(r"[a-zA-Z0-9$]+")
HYDE_BASE_URL = "https://api.vectorengine.ai/v1"
HYDE_WEB_SEARCH_PROMPT = """Please write a passage to answer the question.
Question: {}
Passage:"""


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

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score_tokens(self, query_tokens: list[str], doc_idx: int) -> float:
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
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-12))
            score += idf * (freq * (self.k1 + 1) / (denom + 1e-12))
        return score


RankFn = Callable[[str], list[int]]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_bm25_ranker(corpus: list[dict]) -> RankFn:
    tokenized_docs = [tokenize(row["text"]) for row in corpus]
    bm25 = BM25(tokenized_docs)

    def rank(query: str) -> list[int]:
        query_tokens = tokenize(query)
        scores = [bm25.score_tokens(query_tokens, i) for i in range(len(corpus))]
        return list(np.argsort(-np.asarray(scores, dtype=np.float32)))

    return rank


def build_dense_ranker(model_name: str, corpus: list[dict], query_prefix: str = "") -> RankFn:
    model = load_sentence_encoder(model_name)
    doc_texts = [row["text"] for row in corpus]
    doc_emb = model.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

    def rank(query: str) -> list[int]:
        formatted_query = f"{query_prefix}{query}" if query_prefix else query
        q_emb = model.encode([formatted_query], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = np.dot(doc_emb, q_emb)
        return list(np.argsort(-scores))

    return rank


class HydeOpenAIGenerator:
    """OpenAI-compatible generator aligned with texttron/hyde defaults.

    Original repository defaults:
    - n=8
    - max_tokens=512
    - temperature=0.7
    - top_p=1
    - stop=["\n\n\n"]

    Here we keep the same exposed defaults but issue one request per sample for
    compatibility with GPT-style chat APIs that may not reliably support n>1 or
    completion logprobs in the same way as the original codebase.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str = HYDE_BASE_URL,
        n: int = 8,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        wait_till_success: bool = False,
        request_timeout: float = 120.0,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop if stop is not None else ["\n\n\n"]
        self.wait_till_success = wait_till_success
        self.request_timeout = request_timeout
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.request_timeout)

    def generate(self, prompt: str) -> list[str]:
        outputs: list[str] = []
        while len(outputs) < self.n:
            try:
                result = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stop=self.stop,
                )
                text = (result.choices[0].message.content or "").strip()
                if text:
                    outputs.append(text)
                else:
                    outputs.append("")
            except Exception:
                if not self.wait_till_success:
                    raise
                time.sleep(1.0)
        return outputs


class HydePromptor:
    def __init__(self, task: str = "web search"):
        if task != "web search":
            raise ValueError(f"Unsupported HyDE task: {task}")
        self.task = task

    def build_prompt(self, query: str) -> str:
        return HYDE_WEB_SEARCH_PROMPT.format(query)


class HydeCache:
    def __init__(self, cache_path: str | Path | None = None):
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            print(f"Loading HyDE cache from {self.cache_path}...")
            with self.cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    query = row.get("query")
                    if query:
                        self._cache[query] = row

    def get(self, query: str) -> dict | None:
        return self._cache.get(query)

    def set(self, query: str, payload: dict) -> None:
        self._cache[query] = payload
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _require_pyserini_auto_query_encoder() -> type:
    if AutoQueryEncoder is None:
        raise ImportError(
            "Pyserini is required for the HyDE ranker in official-encoder mode. "
            "Install it with `pip install pyserini` and ensure Java is available."
        )
    return AutoQueryEncoder


def build_hyde_ranker(
    corpus: list[dict],
    encoder_name: str,
    api_key: str,
    generator_model_name: str,
    cache_path: str | Path | None = None,
    hyde_task: str = "web search",
    hyde_n: int = 8,
    hyde_max_tokens: int = 512,
    hyde_temperature: float = 0.7,
    hyde_top_p: float = 1.0,
    hyde_stop: list[str] | None = None,
    wait_till_success: bool = False,
) -> RankFn:
    """Build a HyDE ranker aligned with texttron/hyde under the local-corpus motivation setup.
    """

    auto_query_encoder_cls = _require_pyserini_auto_query_encoder()
    resolved_encoder_name = resolve_local_model_path(encoder_name)
    encoder = auto_query_encoder_cls(encoder_dir=resolved_encoder_name, pooling="mean")
    doc_texts = [row["text"] for row in corpus]
    doc_emb = np.asarray([np.asarray(encoder.encode(text), dtype=np.float32) for text in doc_texts], dtype=np.float32)

    promptor = HydePromptor(task=hyde_task)
    generator = HydeOpenAIGenerator(
        api_key=api_key,
        model_name=generator_model_name,
        n=hyde_n,
        max_tokens=hyde_max_tokens,
        temperature=hyde_temperature,
        top_p=hyde_top_p,
        stop=hyde_stop,
        wait_till_success=wait_till_success,
    )
    cache = HydeCache(cache_path)

    def rank(query: str) -> list[int]:
        cached = cache.get(query)
        if cached is None:
            prompt = promptor.build_prompt(query)
            hypothetical_docs = generator.generate(prompt)
            cached = {
                "query": query,
                "prompt": prompt,
                "hypothetical_documents": hypothetical_docs,
                "generator": {
                    "model_name": generator_model_name,
                    "base_url": HYDE_BASE_URL,
                    "n": hyde_n,
                    "max_tokens": hyde_max_tokens,
                    "temperature": hyde_temperature,
                    "top_p": hyde_top_p,
                    "stop": hyde_stop if hyde_stop is not None else ["\n\n\n"],
                },
            }
            cache.set(query, cached)

        hypothesis_documents = list(cached.get("hypothetical_documents", []))
        encode_inputs = [query] + hypothesis_documents
        emb = np.asarray([np.asarray(encoder.encode(text), dtype=np.float32) for text in encode_inputs], dtype=np.float32)
        q_emb = np.mean(emb, axis=0)
        scores = np.dot(doc_emb, q_emb)
        return list(np.argsort(-scores))

    return rank
