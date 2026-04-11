from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Callable

import numpy as np

from experiments.common import load_sentence_encoder

TOKEN_RE = re.compile(r"[a-zA-Z0-9$]+")


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
