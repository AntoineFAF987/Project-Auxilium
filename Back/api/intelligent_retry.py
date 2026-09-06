"""Deterministic, bounded pre-generation retrieval retry.

This module deliberately does not call an LLM.  It turns the already computed
evidence/answerability signals into a small, auditable retrieval gap and one or
two complementary queries.  Keeping this separate from the retrieval engine
means retry queries still use the normal hybrid/FAISS/BM25/MMR/reranker path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Iterable, Mapping, Sequence

from rag_core.context_sufficiency import AnswerabilityDecision
from .multi_query_retrieval import _canonical


_GLUE = re.compile(r"\s+(?:and|et|ainsi que|,|;)+\s+", re.IGNORECASE)
_QUESTION_WORDS = re.compile(
    r"^(?:le|la|les|un|une|du|de la|des|est[- ]il|sont[- ]elles|qui|quand|quelle?s?|what|which|who|when)\s+",
    re.IGNORECASE,
)


def specific_anchors(text: str) -> list[str]:
    """Generic exact identifiers, preserving their original spelling/order."""
    found: list[str] = []
    for token in re.findall(r"\b(?:[A-Za-z]+[A-Za-z0-9-]*\d[A-Za-z0-9-]*|\d+(?:\.\d+)+|\d{2,})\b", text or ""):
        if token.casefold() not in {item.casefold() for item in found}:
            found.append(token)
    return found


def _terms(text: str) -> set[str]:
    return {
        term for term in _canonical(text).split()
        if len(term) > 2 and not any(char.isdigit() for char in term)
    }


def _aspects(query: str) -> list[str]:
    """Keep user phrasing; split only explicit multi-part question structure."""
    parts = [part.strip(" ?.:!") for part in _GLUE.split(query or "")]
    cleaned = [_QUESTION_WORDS.sub("", part).strip() for part in parts]
    return [part for part in cleaned if _terms(part)] or [query.strip()]


@dataclass(frozen=True)
class RetrievalGap:
    missing_aspects: list[str]
    supported_aspects: list[str]
    anchors: list[str]
    reason: str
    answerability_before_retry: str
    evidence_mode: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_retrieval_gap(
    *, query: str, context_text: str, answerability: AnswerabilityDecision,
    evidence_mode: str, evidence_reason: str = "",
) -> RetrievalGap:
    """Derive missing facets from final evidence without another judge."""
    anchors = specific_anchors(query)
    context_terms = _terms(context_text)
    supported, missing = [], []
    for aspect in _aspects(query):
        aspect_terms = _terms(aspect)
        # An aspect is only claimed supported when its descriptive vocabulary
        # is represented in final evidence. Identifiers are checked separately.
        if aspect_terms and aspect_terms.issubset(context_terms):
            supported.append(aspect)
        else:
            missing.append(aspect)
    unsupported_anchors = [anchor for anchor in anchors if anchor.casefold() not in {item.casefold() for item in answerability.supported_anchors}]
    for anchor in unsupported_anchors:
        marker = f"exact technical information for {anchor}"
        if marker not in missing:
            missing.insert(0, marker)
    if not missing and answerability.status != "answerable":
        missing = [query.strip()]
    reason = "missing_anchor" if unsupported_anchors else (
        "partial_answerability" if answerability.status == "partial" else "unanswerable"
    )
    if evidence_reason:
        reason = f"{reason}:{evidence_reason}"
    return RetrievalGap(
        missing_aspects=missing,
        supported_aspects=supported,
        anchors=anchors,
        reason=reason,
        answerability_before_retry=answerability.status,
        evidence_mode=evidence_mode,
        confidence=answerability.context_sufficiency.confidence,
    )


def _near_duplicate(left: str, right: str) -> bool:
    left_terms, right_terms = set(_canonical(left).split()), set(_canonical(right).split())
    if not left_terms or not right_terms or _canonical(left) == _canonical(right):
        return True
    return left_terms == right_terms or len(left_terms & right_terms) / len(left_terms | right_terms) >= 0.82


def build_retry_queries(
    *, original_user_query: str, orchestrator_query: str | None,
    resolved_retrieval_query: str | None, gap: RetrievalGap,
    max_queries: int = 2,
) -> tuple[list[str], list[dict[str, str]]]:
    """Build complementary searches from *missing* aspects only.

    Exact anchors are prepended to every query.  A resolved follow-up subject
    is used as a stable qualifier, never the raw conversational follow-up.
    """
    del orchestrator_query  # The resolved query already contains its subject.
    base_queries = [original_user_query, resolved_retrieval_query or ""]
    accepted: list[str] = []
    rejected: list[dict[str, str]] = []
    anchor_prefix = " ".join(gap.anchors)
    for aspect in gap.missing_aspects:
        # Do not turn the retry into a paraphrase of a fully supported facet.
        aspect_text = aspect.strip()
        candidate = " ".join(part for part in (anchor_prefix, aspect_text) if part).strip()
        if not candidate:
            continue
        if any(_near_duplicate(candidate, previous) for previous in base_queries + accepted):
            rejected.append({"query": candidate, "reason": "near_duplicate_of_initial_or_retry"})
            continue
        # Anchors from the original request are inviolable, even if evidence
        # happened to contain a neighbouring reference.
        if any(anchor.casefold() not in candidate.casefold() for anchor in gap.anchors):
            rejected.append({"query": candidate, "reason": "missing_preserved_anchor"})
            continue
        accepted.append(candidate[:400])
        if len(accepted) >= max_queries:
            break
    return accepted, rejected


def merge_cumulative_evidence(
    initial: Sequence[tuple[float, Mapping[str, Any]]], retry: Sequence[tuple[float, Mapping[str, Any]]],
    *, retry_query_count: int,
) -> list[tuple[float, dict[str, Any]]]:
    """UID-deduplicated cumulative pool retaining first-pass score/provenance."""
    merged: dict[str, tuple[float, dict[str, Any]]] = {}
    for round_number, rows in ((1, initial), (2, retry)):
        for score, raw_meta in rows:
            meta = dict(raw_meta)
            uid = str(meta.get("chunk_uid") or f"{meta.get('document_id')}:{meta.get('chunk_id')}")
            current = merged.get(uid)
            prior_by = list((current[1].get("retrieved_by") if current else []) or [])
            current_by = list(meta.get("retrieved_by") or [])
            if round_number == 1:
                current_by = ["initial", *current_by]
            elif not current_by:
                # Single-query retry rows do not pass through RRF, so give
                # them a precise fallback provenance marker.
                current_by = ["retry_1" if retry_query_count == 1 else "retry"]
            provenance = list(dict.fromkeys([*prior_by, *current_by]))
            if current:
                retained_score, retained = current
                retained["retrieved_by"] = provenance
                retained["retrieval_rounds"] = sorted(set((retained.get("retrieval_rounds") or [1]) + [round_number]))
                retained.setdefault("initial_score", retained_score)
                merged[uid] = (retained_score, retained)
            else:
                meta["retrieved_by"] = provenance
                meta["retrieval_rounds"] = [round_number]
                meta["initial_score"] = float(score) if round_number == 1 else None
                merged[uid] = (float(score), meta)
    return list(merged.values())
