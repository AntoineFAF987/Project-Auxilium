"""Deterministic query variants and rank fusion before CandidatePool creation."""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def _canonical(value: str) -> str:
    text = unicodedata.normalize("NFKD", value.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _obviously_duplicate(left: str, right: str) -> bool:
    """Deduplicate only safe textual equivalents; preserve useful rewrites."""
    left_canonical, right_canonical = _canonical(left), _canonical(right)
    if left_canonical == right_canonical:
        return True
    return bool(left_canonical and set(left_canonical.split()) == set(right_canonical.split()))


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
    """Keep only structured documentary refinements from a follow-up.

    The orchestrator has already resolved the subject. Free prose in a
    follow-up is an instruction to the assistant, not a search query. We
    retain generic structural refinements only: reference-like identifiers,
    a named entity, and an explicit email/date locator.
    """
    subject_terms = set(_canonical(subject).split())
    words = re.findall(r"[\wÀ-ÿ0-9-]+", raw_user_message)
    additions: list[str] = []
    for index, word in enumerate(words):
        canonical = _canonical(word)
        if not canonical or canonical in subject_terms:
            continue
        if any(char.isdigit() for char in word):
            additions.append(word)
        elif index > 0 and re.fullmatch(r"[A-ZÀ-Ö][A-Za-zÀ-ÿ'-]{2,}", word):
            additions.append(word)
    lowered = [_canonical(word) for word in words]
    for index, token in enumerate(lowered):
        if token not in {"mail", "email", "courriel"}:
            continue
        window = words[max(0, index - 2):index + 4]
        if any(any(char.isdigit() for char in value) for value in window):
            additions.extend(value for value in window if any(char.isdigit() for char in value))
            additions.append(words[index])
    return " ".join(dict.fromkeys(additions))


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
    """Build factual query variants independently of any response presentation.

    An autonomous user query is retained because it can contain exact anchors
    lost by a rewrite. For a resolved conversational follow-up, the raw turn
    is preserved in history and logs but is not itself a documentary query.
    """
    # The user's wording is evidence too: it often contains an exact model,
    # reference, or relation that a planner paraphrase can accidentally lose.
    # A resolved follow-up is additive, never a silent replacement for it.
    standalone = resolved_query or orchestrator_query
    options: list[tuple[str, str | None]] = (
        [
            ("resolved_followup", standalone),
            ("normalized", normalized_query(standalone or original_query)),
        ]
        if follow_up else [
            ("original_autonomous", original_query),
            ("orchestrator", standalone),
            ("normalized", normalized_query(standalone or original_query)),
        ]
    )
    result: list[tuple[str, str]] = []
    for kind, query in options:
        if not query or len(query.strip()) < 3:
            continue
        if any(_obviously_duplicate(query, existing) for _, existing in result):
            continue
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
