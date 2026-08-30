"""Deterministic query variants and rank fusion before CandidatePool creation."""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def _canonical(value: str) -> str:
    text = unicodedata.normalize("NFKD", value.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def normalized_query(original: str) -> str | None:
    """Remove conversational glue only; never add or substitute domain terms."""
    filler = {
        "je", "me", "m", "demandais", "demande", "si", "j", "ai", "avais", "enfin",
        "peux", "tu", "vous", "pourrais", "voudrais", "savoir", "juste", "svp", "please",
        "can", "you", "could", "would", "i", "do", "does", "the", "a", "an",
    }
    words = re.findall(r"[\wÀ-ÿ0-9-]+", original)
    compact = [word for word in words if _canonical(word) not in filler]
    candidate = " ".join(compact).strip()
    if len(compact) < 3 or _canonical(candidate) == _canonical(original):
        return None
    return candidate[:400]


def build_retrieval_queries(*, original_query: str, orchestrator_query: str | None) -> list[tuple[str, str]]:
    """At most three distinct representations, original always first."""
    options: list[tuple[str, str | None]] = [
        ("original", original_query),
        ("normalized", normalized_query(original_query)),
        ("orchestrator", orchestrator_query),
    ]
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for kind, query in options:
        if not query or len(query.strip()) < 3:
            continue
        key = _canonical(query)
        if key in seen:
            continue
        seen.add(key)
        result.append((kind, query.strip()))
    return result[:3]


def reciprocal_rank_fusion(
    rankings: list[tuple[str, list[tuple[float, dict[str, Any]]]]], *, k: int = 60,
) -> list[tuple[float, dict[str, Any]]]:
    """Fuse query-specific rankings without comparing their raw score scales."""
    merged: dict[str, tuple[float, dict[str, Any], dict[str, int], dict[str, float]]] = {}
    for query_kind, rows in rankings:
        for rank, (score, meta) in enumerate(rows, start=1):
            uid = str(meta.get("chunk_uid") or f"{meta.get('document_id')}:{meta.get('chunk_id')}")
            fusion, best_meta, ranks, scores = merged.get(uid, (0.0, dict(meta), {}, {}))
            ranks[query_kind] = rank
            scores[query_kind] = float(score)
            merged[uid] = (fusion + 1.0 / (k + rank), best_meta, ranks, scores)
    result = []
    for fusion, meta, ranks, scores in merged.values():
        enriched = dict(meta)
        enriched["retrieved_by"] = list(ranks)
        enriched["per_query_rank"] = ranks
        enriched["per_query_score"] = scores
        enriched["fusion_score"] = fusion
        result.append((fusion, enriched))
    return sorted(result, key=lambda item: item[0], reverse=True)
