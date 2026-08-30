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


def _followup_additions(raw_user_message: str, subject: str) -> str:
    """Keep new subject qualifiers while discarding conversational search glue."""
    glue = {
        "and", "anything", "dans", "do", "en", "et", "find", "for", "in", "les", "local", "locales", "locaux",
        "mail", "mails", "mes", "my", "nothing", "rien", "source", "sources", "the", "tu", "you",
        "trouves", "trouver", "vos", "your", "can", "could", "check", "documents", "document",
    }
    subject_terms = set(_canonical(subject).split())
    additions = [word for word in re.findall(r"[\wÀ-ÿ0-9-]+", raw_user_message)
                 if _canonical(word) not in glue and _canonical(word) not in subject_terms and len(_canonical(word)) >= 3]
    return " ".join(additions)


def resolve_retrieval_query(*, raw_user_message: str, orchestrator_query: str, history: list[dict[str, Any]] | None = None) -> str:
    """Resolve a follow-up to its subject without a second model invocation.

    The orchestrator's standalone subject is the stable base; the current turn
    can only append explicit, non-conversational qualifiers (for example a name).
    History is intentionally accepted as execution context, while subject
    resolution stays deterministic and does not reinterpret free-form prose.
    """
    del history  # The plan was built from compact history; avoid a second semantic pass here.
    subject = orchestrator_query.strip()
    additions = _followup_additions(raw_user_message, subject)
    return " ".join(part for part in (subject, additions) if part).strip()


def build_retrieval_queries(*, original_query: str, orchestrator_query: str | None, resolved_query: str | None = None, follow_up: bool = False) -> list[tuple[str, str]]:
    """Build contextual variants; literal wording is retained only for autonomous turns."""
    base = resolved_query or orchestrator_query or original_query
    options: list[tuple[str, str | None]] = [
        *(([("resolved_subject", base), ("normalized_resolved_subject", normalized_query(base)), ("orchestrator", orchestrator_query)] if follow_up else
          [("original", original_query), ("normalized", normalized_query(original_query)), ("orchestrator", orchestrator_query)])),
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
