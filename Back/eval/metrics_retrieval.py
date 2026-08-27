from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_KS = (1, 3, 5, 10)


def _deduplicate_ranked(ids: Sequence[str]) -> List[str]:
    seen = set()
    output = []
    for chunk_id in ids:
        if chunk_id not in seen:
            seen.add(chunk_id)
            output.append(chunk_id)
    return output


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = len(set(_deduplicate_ranked(retrieved)[:k]) & relevant_set)
    return hits / len(relevant_set)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    hits = len(set(_deduplicate_ranked(retrieved)[:k]) & relevant_set)
    return hits / k


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    for rank, chunk_id in enumerate(_deduplicate_ranked(retrieved), start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    return float(bool(set(_deduplicate_ranked(retrieved)[:k]) & relevant_set))


def ndcg_at_k(
    retrieved: Sequence[str], relevance_grades: Mapping[str, float], k: int
) -> Optional[float]:
    """Compute graded NDCG using gain ``2**grade - 1``.

    Returns ``None`` when no graded judgments are supplied. This keeps binary
    relevance metrics distinct from explicitly graded NDCG experiments.
    """

    positive_grades = {key: float(value) for key, value in relevance_grades.items() if value > 0}
    if not positive_grades:
        return None

    ranked = _deduplicate_ranked(retrieved)[:k]
    dcg = sum(
        (2.0 ** positive_grades.get(chunk_id, 0.0) - 1.0) / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked, start=1)
    )
    ideal = sorted(positive_grades.values(), reverse=True)[:k]
    idcg = sum(
        (2.0**grade - 1.0) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal, start=1)
    )
    return dcg / idcg if idcg else 0.0


def compute_query_metrics(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    *,
    answerable: bool,
    relevance_grades: Optional[Mapping[str, float]] = None,
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[str, Optional[float]]:
    """Compute retrieval metrics for one query.

    Non-answerable questions have no positive retrieval target. Their metrics
    are therefore ``None`` and are excluded from retrieval averages. Their
    correctness belongs to the future abstention/generation benchmark.
    """

    metric_names = [
        *(f"recall@{k}" for k in ks),
        *(f"precision@{k}" for k in ks),
        *(f"hit_rate@{k}" for k in ks),
        "mrr",
        *(f"ndcg@{k}" for k in ks),
    ]
    if not answerable:
        return {name: None for name in metric_names}

    relevant_set = set(relevant)
    metrics: Dict[str, Optional[float]] = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant_set, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved, relevant_set, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, relevant_set, k)
    metrics["mrr"] = reciprocal_rank(retrieved, relevant_set)
    for k in ks:
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevance_grades or {}, k)
    return metrics


def average_metrics(
    metric_rows: Iterable[Mapping[str, Optional[float]]],
) -> Dict[str, Optional[float]]:
    values = defaultdict(list)
    all_names = set()
    for row in metric_rows:
        all_names.update(row)
        for name, value in row.items():
            if value is not None:
                values[name].append(float(value))
    return {
        name: (sum(values[name]) / len(values[name]) if values[name] else None)
        for name in sorted(all_names)
    }
