"""Deterministic, bounded pre-generation retrieval retry.

This module deliberately does not call an LLM.  It turns the already computed
evidence/answerability signals into a small, auditable retrieval gap and one or
two complementary queries.  Keeping this separate from the retrieval engine
means retry queries still use the normal hybrid/FAISS/BM25/MMR/reranker path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping, Sequence

from rag_core.context_sufficiency import AnswerabilityDecision
from .multi_query_retrieval import _canonical


_FINAL_OUTCOME = re.compile(r"\b(?:approved|approval|rejected|rejection|final|decision|approuv|refus|d[ée]cision|autoris\w*)\b", re.IGNORECASE)
_PENDING = re.compile(r"\b(?:pending|awaiting|submitted|request(?:ed)?|attend|en attente|d[ée]pos[ée]|transmi[se])\b", re.IGNORECASE)
_CURRENT = re.compile(r"\b(?:current|latest|updated|today|actuel|dernier|mise [àa] jour|aujourd)\b", re.IGNORECASE)


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


def extract_requested_aspects(*, query: str, query_semantics: str = "general_document_question") -> list[str]:
    """Extract concise, generic information needs instead of query wording."""
    canonical = _canonical(query)
    aspects: list[str] = []
    # These are information categories, not product/domain vocabulary.
    categories = (
        (("temperature", "temperature"), "operating temperature limits"),
        (("pressure", "pression"), "pressure limits"),
        (("material", "materiau", "matiere"), "material"),
        (("dimension", "diameter", "diametre"), "dimensions"),
        (("compatib",), "compatibility"),
        (("price", "prix", "cost", "cout"), "price"),
        (("delivery", "delai", "lead time"), "delivery time"),
    )
    for markers, aspect in categories:
        if any(marker in canonical for marker in markers):
            aspects.append(aspect)
    if re.search(r"\b(?:nace|certif|certified|certifie)\b", canonical):
        # Preserve an explicitly named certification standard when present;
        # otherwise keep the generic information category.
        aspects.append("NACE certification" if "nace" in canonical else "certification")
    if re.search(r"\b(?:valid\w*|approved by|qui a approuv|who approved)\b", canonical):
        if re.search(r"\b(?:qui|who|identity)\b", canonical):
            aspects.append("validator identity")
        if re.search(r"\b(?:date|quand|when)\b", canonical):
            aspects.append("validation date")
    if query_semantics == "decision" or re.search(r"\b(?:approved|approuv|autorise|decision finale)\b", canonical):
        aspects.append("final decision / approval status")
    if query_semantics == "current_state":
        aspects.append("latest/current state")
    # No fallback to the original query: an unknown aspect is not a safe
    # reason to repeat the initial retrieval.
    return list(dict.fromkeys(aspects))


def infer_supported_aspects(*, requested_aspects: Sequence[str], context_text: str, query_semantics: str) -> list[str]:
    """Deterministically identify supported facets from retained evidence."""
    context = _canonical(context_text)
    supported: list[str] = []
    for aspect in requested_aspects:
        terms = set(_canonical(aspect).split())
        if aspect == "NACE certification":
            is_supported = "nace" in context
        elif aspect == "operating temperature limits":
            is_supported = "temperature" in context
        elif aspect == "pressure limits":
            is_supported = "pressure" in context or "pression" in context
        elif aspect == "final decision / approval status":
            # Pending/request-submitted evidence describes the subject but is
            # explicitly insufficient to support an outcome.
            is_supported = bool(_FINAL_OUTCOME.search(context_text)) and not bool(_PENDING.search(context_text) and not re.search(r"\b(?:approved|rejected|approuv|refus|final|autoris\w*)\b", context, re.I))
        elif aspect == "latest/current state":
            is_supported = bool(_CURRENT.search(context_text)) and not bool(_PENDING.search(context_text))
        else:
            is_supported = bool(terms) and terms.issubset(set(context.split()))
        if is_supported:
            supported.append(aspect)
    # These evidence-state facts aid auditability for decision questions, but
    # never masquerade as the requested final outcome.
    if query_semantics in {"decision", "current_state"}:
        if re.search(r"\b(?:submitted|request|depos|demand)\b", context):
            supported.append("request submitted")
        if _PENDING.search(context_text):
            supported.append("decision pending")
    return list(dict.fromkeys(supported))


@dataclass(frozen=True)
class AspectCoverage:
    requested_aspects: list[str]
    supported_aspects: list[str]
    missing_aspects: list[str]
    gap_derivation_method: str


def evaluate_aspect_coverage(*, query: str, context_text: str, query_semantics: str) -> AspectCoverage:
    """Single deterministic aspect signal shared by retry and answerability."""
    requested = extract_requested_aspects(query=query, query_semantics=query_semantics)
    supported = infer_supported_aspects(
        requested_aspects=requested, context_text=context_text, query_semantics=query_semantics,
    )
    missing = [aspect for aspect in requested if aspect not in supported]
    method = "requested_minus_supported"
    if query_semantics == "decision" and "final decision / approval status" in missing:
        method = "decision_semantics_missing_final_outcome"
    elif query_semantics == "current_state" and "latest/current state" in missing:
        method = "current_state_semantics_missing_latest_state"
    return AspectCoverage(requested, supported, missing, method)


@dataclass(frozen=True)
class RetrievalGap:
    requested_aspects: list[str]
    missing_aspects: list[str]
    supported_aspects: list[str]
    anchors: list[str]
    reason: str
    answerability_before_retry: str
    evidence_mode: str
    confidence: float
    gap_derivation_method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_retrieval_gap(
    *, query: str, context_text: str, answerability: AnswerabilityDecision,
    evidence_mode: str, evidence_reason: str = "", query_semantics: str = "general_document_question",
) -> RetrievalGap:
    """Derive missing facets from final evidence without another judge."""
    anchors = specific_anchors(query)
    aspect_coverage = evaluate_aspect_coverage(
        query=query, context_text=context_text, query_semantics=query_semantics,
    )
    requested, supported, missing = (
        aspect_coverage.requested_aspects,
        aspect_coverage.supported_aspects,
        aspect_coverage.missing_aspects,
    )
    unsupported_anchors = [anchor for anchor in anchors if anchor.casefold() not in {item.casefold() for item in answerability.supported_anchors}]
    # Anchors stay separate from aspects. A missing exact reference is enough
    # to trigger the existing anchor protection, but never copied into a gap.
    reason = "missing_anchor" if unsupported_anchors else (
        "partial_answerability" if answerability.status == "partial" else "unanswerable"
    )
    if evidence_reason:
        reason = f"{reason}:{evidence_reason}"
    return RetrievalGap(
        requested_aspects=requested,
        missing_aspects=missing,
        supported_aspects=supported,
        anchors=anchors,
        reason=reason,
        answerability_before_retry=answerability.status,
        evidence_mode=evidence_mode,
        confidence=answerability.context_sufficiency.confidence,
        gap_derivation_method=aspect_coverage.gap_derivation_method,
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
    base_queries = [original_user_query, resolved_retrieval_query or ""]
    accepted: list[str] = []
    rejected: list[dict[str, str]] = []
    anchor_prefix = " ".join(gap.anchors)
    # A non-technical decision/state question still needs its resolved subject
    # (for example an email thread) beside the missing outcome category.
    subject_prefix = "" if anchor_prefix else (resolved_retrieval_query or orchestrator_query or "").strip()
    for aspect in gap.missing_aspects:
        # Do not turn the retry into a paraphrase of a fully supported facet.
        aspect_text = aspect.strip()
        candidate = " ".join(part for part in (anchor_prefix or subject_prefix, aspect_text) if part).strip()
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
