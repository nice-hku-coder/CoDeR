from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(ranked_doc_ids: Sequence[str], relevant_doc_ids: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0
    topk = ranked_doc_ids[:k]
    hit = sum(1 for doc_id in topk if doc_id in relevant)
    return hit / len(relevant)


def precision_at_k(ranked_doc_ids: Sequence[str], positive_doc_ids: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    positives = set(positive_doc_ids)
    topk = ranked_doc_ids[:k]
    if not topk:
        return 0.0
    hit = sum(1 for doc_id in topk if doc_id in positives)
    return hit / len(topk)


def ccr_at_k(ranked_doc_ids: Sequence[str], constraint_satisfying_doc_ids: Iterable[str], k: int) -> float:
    # CCR is top-k precision on constraint-satisfying labels.
    return precision_at_k(ranked_doc_ids, constraint_satisfying_doc_ids, k)


def ndcg_at_k(
    ranked_doc_ids: Sequence[str],
    graded_relevance: dict[str, float],
    k: int,
) -> float:
    if k <= 0:
        return 0.0

    def dcg(doc_ids: Sequence[str]) -> float:
        score = 0.0
        for i, doc_id in enumerate(doc_ids[:k], start=1):
            rel = float(graded_relevance.get(doc_id, 0.0))
            if rel > 0.0:
                score += rel / math.log2(i + 1)
        return score

    actual = dcg(ranked_doc_ids)
    ideal_docs = sorted(graded_relevance.keys(), key=lambda d: graded_relevance[d], reverse=True)
    ideal = dcg(ideal_docs)
    if ideal <= 1e-12:
        return 0.0
    return actual / ideal

