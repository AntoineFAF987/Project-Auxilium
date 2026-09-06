from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from time import perf_counter
from typing import Any, Dict, List, Literal, Mapping, Sequence, Set, Tuple
import unicodedata

from .constants import FINAL_K


# Deliberately small, benchmark-first policy. 0.5 is the probability boundary
# of the current reranker; the other thresholds must be recalibrated on a
# larger, representative benchmark before production activation.
STRONG_RERANKER_SCORE = 0.5
MIN_QUERY_TERM_COVERAGE = 0.55
REDUNDANCY_JACCARD = 0.82
WEAK_EVIDENCE_PATIENCE = 3
WEAK_EVIDENCE_SCORE = 0.15

_STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du",
    "et", "est", "elle", "en", "la", "le", "les", "leur", "leurs", "ou",
    "par", "pour", "que", "quel", "quelle", "quelles", "quels", "qui",
    "quoi", "se", "son", "sur", "the", "what", "which", "who", "why",
    "with", "and", "for", "from", "into", "all", "are", "was", "were",
}


@dataclass(frozen=True)
class SufficiencyPolicy:
    strong_reranker_score: float = STRONG_RERANKER_SCORE
    min_query_term_coverage: float = MIN_QUERY_TERM_COVERAGE
    redundancy_jaccard: float = REDUNDANCY_JACCARD
    weak_evidence_patience: int = WEAK_EVIDENCE_PATIENCE
    weak_evidence_score: float = WEAK_EVIDENCE_SCORE


@dataclass(frozen=True)
class QueryEvidenceIntent:
    kind: str
    minimum_evidence: int
    minimum_documents: int
    matched_markers: Tuple[str, ...]


@dataclass(frozen=True)
class SufficiencyDecision:
    sufficient: bool
    reason: str
    confidence: float
    requested_more_evidence: int
    selected_count: int
    distinct_documents: int
    query_term_coverage: float
    strong_top_evidence: bool
    protected_evidence_missing: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


Answerability = Literal["answerable", "partial", "unanswerable"]


@dataclass(frozen=True)
class AnswerabilityDecision:
    """Production-facing decision made after retrieval has produced final blocks.

    This is intentionally deterministic.  It does not pretend that retrieval
    scores prove an answer; scores only complement topical relevance, direct
    technical-anchor support, coverage, and structural evidence sufficiency.
    """

    status: Answerability
    reasons: Tuple[str, ...]
    query_term_coverage: float
    requested_anchors: Tuple[str, ...]
    supported_anchors: Tuple[str, ...]
    context_sufficiency: SufficiencyDecision

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answerability": self.status,
            "answerability_reasons": list(self.reasons),
            "query_term_coverage": self.query_term_coverage,
            "requested_anchors": list(self.requested_anchors),
            "supported_anchors": list(self.supported_anchors),
            "context_sufficiency": self.context_sufficiency.to_dict(),
        }


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _tokens(value: str) -> Set[str]:
    return {
        token for token in re.findall(r"[a-z0-9][a-z0-9_.-]*", _normalize(value))
        if len(token) > 1 and token not in _STOPWORDS
    }


def _technical_anchors(value: str) -> Set[str]:
    """Extract generic identifier-like terms, without a product vocabulary."""
    return {
        token.casefold()
        for token in re.findall(r"\b(?=[\w-]*\d)[\w-]{2,}\b", value or "")
    }


def classify_evidence_intent(query: str) -> QueryEvidenceIntent:
    normalized = _normalize(query)
    comparison = tuple(
        marker for marker in ("compare", "comparaison", "difference", "versus", " vs ")
        if marker in normalized
    )
    exhaustive = tuple(
        marker for marker in ("enumere", "tous les", "toutes les", "liste complete", "all the", "list all")
        if marker in normalized
    )
    interrogatives = sum(
        len(re.findall(pattern, normalized))
        for pattern in (
            r"\bquel(?:le|les|s)?\b", r"\bcombien\b", r"\bpourquoi\b",
            r"\bcomment\b", r"\bquand\b", r"\bqui\b", r"\bwhat\b",
            r"\bwhich\b", r"\bwhy\b", r"\bhow\b", r"\bwhen\b",
        )
    )
    if comparison:
        return QueryEvidenceIntent("comparison", 2, 2, comparison)
    if exhaustive:
        return QueryEvidenceIntent("exhaustive", 3, 1, exhaustive)
    if interrogatives >= 2 or " ainsi que " in f" {normalized} ":
        return QueryEvidenceIntent("multi_fact", 2, 1, ("multiple_clauses",))
    return QueryEvidenceIntent("single", 1, 1, ())


def _document_id(meta: Mapping[str, Any]) -> str:
    return str(meta.get("document_id") or meta.get("path") or meta.get("file") or "unknown")


def _signal_score(signal: Mapping[str, Any]) -> float:
    for name in ("reranker_score", "hybrid_score", "dense_score", "bm25_score"):
        if signal.get(name) is not None:
            return float(signal[name])
    return 0.0


def _rank_agreement(signal: Mapping[str, Any]) -> bool:
    dense_rank, bm25_rank = signal.get("dense_rank"), signal.get("bm25_rank")
    return dense_rank is not None and bm25_rank is not None and dense_rank <= 3 and bm25_rank <= 3


def _coverage(query_terms: Set[str], selected_texts: Sequence[str]) -> float:
    if not query_terms:
        return 1.0
    evidence_terms: Set[str] = set()
    for text in selected_texts:
        evidence_terms.update(_tokens(text))
    return len(query_terms & evidence_terms) / len(query_terms)


def evaluate_context_sufficiency(
    *,
    query: str,
    selected_ids: Sequence[int],
    metas: Sequence[Mapping[str, Any]],
    texts: Sequence[str],
    candidate_signals: Mapping[int, Mapping[str, Any]],
    protected_candidate_ids: Set[int] | None = None,
    intent: QueryEvidenceIntent | None = None,
    policy: SufficiencyPolicy = SufficiencyPolicy(),
) -> SufficiencyDecision:
    intent = intent or classify_evidence_intent(query)
    protected = protected_candidate_ids or set()
    selected_set = set(selected_ids)
    missing_protected = len(protected - selected_set)
    documents = {_document_id(metas[idx]) for idx in selected_ids}
    coverage = _coverage(_tokens(query), [texts[idx] for idx in selected_ids])
    top_signal = candidate_signals.get(selected_ids[0], {}) if selected_ids else {}
    top_score = _signal_score(top_signal)
    top_exact = bool(float(top_signal.get("exact_match_bonus") or 0.0))
    strong_top = top_exact or top_score >= policy.strong_reranker_score
    coherent_top = top_exact or _rank_agreement(top_signal) or coverage >= policy.min_query_term_coverage

    if missing_protected:
        sufficient, reason = False, "protected_anchor_scope_evidence_missing"
    elif len(selected_ids) < intent.minimum_evidence:
        sufficient, reason = False, f"{intent.kind}_requires_more_evidence"
    elif len(documents) < intent.minimum_documents:
        sufficient, reason = False, f"{intent.kind}_requires_distinct_documents"
    elif intent.kind == "single" and len(selected_ids) == 1 and strong_top and coherent_top:
        sufficient, reason = True, "strong_coherent_single_evidence"
    elif intent.kind != "single" and strong_top:
        sufficient, reason = True, "complex_intent_minimum_evidence_met"
    elif coverage >= policy.min_query_term_coverage and (strong_top or len(selected_ids) >= 2):
        sufficient, reason = True, "intent_and_query_terms_covered"
    elif (
        intent.kind == "single"
        and top_score <= policy.weak_evidence_score
        and len(selected_ids) >= policy.weak_evidence_patience
    ):
        # Avoid filling the prompt with ten weak passages for likely-unanswerable
        # queries. This is deliberately reported as low-confidence sufficiency.
        sufficient, reason = True, "weak_evidence_patience_exhausted"
    else:
        sufficient, reason = False, "insufficient_signal_or_coverage"

    confidence = min(1.0, max(0.0, 0.55 * float(strong_top) + 0.45 * coverage))
    requested = 0 if sufficient else max(
        1,
        intent.minimum_evidence - len(selected_ids),
        intent.minimum_documents - len(documents),
        missing_protected,
    )
    return SufficiencyDecision(
        sufficient=sufficient,
        reason=reason,
        confidence=confidence,
        requested_more_evidence=requested,
        selected_count=len(selected_ids),
        distinct_documents=len(documents),
        query_term_coverage=coverage,
        strong_top_evidence=strong_top,
        protected_evidence_missing=missing_protected,
    )


def _is_redundant(
    candidate_id: int,
    selected_ids: Sequence[int],
    texts: Sequence[str],
    threshold: float,
) -> Tuple[bool, float]:
    candidate_terms = _tokens(texts[candidate_id])
    maximum = 0.0
    for selected_id in selected_ids:
        selected_terms = _tokens(texts[selected_id])
        union = candidate_terms | selected_terms
        similarity = len(candidate_terms & selected_terms) / len(union) if union else 1.0
        maximum = max(maximum, similarity)
    return maximum >= threshold, maximum


def select_adaptive_context(
    *,
    query: str,
    candidates: Sequence[Tuple[float, int]],
    metas: Sequence[Mapping[str, Any]],
    texts: Sequence[str],
    candidate_signals: Mapping[int, Mapping[str, Any]],
    protected_candidate_ids: Sequence[int] = (),
    final_k: int = FINAL_K,
    policy: SufficiencyPolicy = SufficiencyPolicy(),
) -> Tuple[List[Tuple[float, int]], Dict[str, Any]]:
    """Progressively select useful evidence until sufficient or ``final_k``."""
    started = perf_counter()
    pool = list(candidates[:final_k])
    if not pool:
        return [], {
            "candidate_pool_ids": [], "selected_candidate_ids": [],
            "skipped_redundant": [], "decisions": [], "final_decision": None,
            "timing_ms": (perf_counter() - started) * 1000.0,
        }

    intent = classify_evidence_intent(query)
    pool_ids = {idx for _, idx in pool}
    protected = set(protected_candidate_ids) & pool_ids
    selected: List[Tuple[float, int]] = [pool[0]]
    selected_ids = [pool[0][1]]
    decisions: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    def evaluate() -> SufficiencyDecision:
        decision = evaluate_context_sufficiency(
            query=query,
            selected_ids=selected_ids,
            metas=metas,
            texts=texts,
            candidate_signals=candidate_signals,
            protected_candidate_ids=protected,
            intent=intent,
            policy=policy,
        )
        decisions.append(decision.to_dict())
        return decision

    decision = evaluate()
    for score, candidate_id in pool[1:]:
        if decision.sufficient or len(selected) >= final_k:
            break
        redundant, similarity = _is_redundant(
            candidate_id, selected_ids, texts, policy.redundancy_jaccard
        )
        if redundant and candidate_id not in protected:
            skipped.append({
                "candidate_id": candidate_id,
                "reason": "lexical_redundancy",
                "max_jaccard": similarity,
            })
            continue
        selected.append((score, candidate_id))
        selected_ids.append(candidate_id)
        decision = evaluate()

    return selected, {
        "policy": asdict(policy),
        "intent": asdict(intent),
        "candidate_pool_ids": [idx for _, idx in pool],
        "protected_candidate_ids": sorted(protected),
        "selected_candidate_ids": selected_ids,
        "skipped_redundant": skipped,
        "decisions": decisions,
        "final_decision": decision.to_dict(),
        "stopped_before_final_k": len(selected) < min(final_k, len(pool)),
        "timing_ms": (perf_counter() - started) * 1000.0,
    }


def evaluate_answerability(
    *,
    query: str,
    evidence_mode: Literal["direct", "related", "none"],
    context_is_relevant: bool,
    evidence_sufficient: bool | None,
    reranker_accepted: bool,
    blocks: Sequence[Mapping[str, Any]],
    policy: SufficiencyPolicy = SufficiencyPolicy(),
) -> AnswerabilityDecision:
    """Classify final local evidence as answerable, partial, or unanswerable.

    ``blocks`` must be the final, post-enrichment context blocks.  A related
    reference can be useful, but is never enough to mark a precise request as
    answerable by itself.  This protects version/model identifiers generically.
    """
    texts = [str(block.get("text") or "") for block in blocks]
    metas = [dict(block) for block in blocks]
    requested = _technical_anchors(query)
    documented = _technical_anchors("\n".join(texts))
    supported = requested & documented
    signals = {
        index: {"reranker_score": policy.strong_reranker_score if reranker_accepted else 0.0}
        for index in range(len(texts))
    }
    context = evaluate_context_sufficiency(
        query=query,
        selected_ids=list(range(len(texts))),
        metas=metas,
        texts=texts,
        candidate_signals=signals,
        policy=policy,
    ) if texts else SufficiencyDecision(
        sufficient=False, reason="no_final_blocks", confidence=0.0,
        requested_more_evidence=1, selected_count=0, distinct_documents=0,
        query_term_coverage=0.0, strong_top_evidence=False,
        protected_evidence_missing=0,
    )

    reasons: list[str] = []
    if not texts:
        reasons.append("no_final_context_blocks")
    if evidence_mode == "none":
        reasons.append("no_usable_topical_evidence")
    if not context_is_relevant:
        reasons.append("context_not_relevant_to_information_need")
    missing_anchors = requested - documented
    if missing_anchors:
        reasons.append("specific_anchors_not_directly_supported:" + ",".join(sorted(missing_anchors)))
    if evidence_mode == "related":
        reasons.append("only_related_evidence_for_requested_entity")
    if context.query_term_coverage < policy.min_query_term_coverage:
        reasons.append("insufficient_query_term_coverage")
    if evidence_sufficient is False:
        reasons.append("structural_evidence_incomplete")
    if not reranker_accepted:
        reasons.append("reranker_guard_rejected_context")

    if not texts or evidence_mode == "none" or not context_is_relevant:
        status: Answerability = "unanswerable"
    elif evidence_mode == "related" or missing_anchors or evidence_sufficient is False:
        status = "partial"
    elif context.query_term_coverage < policy.min_query_term_coverage or not reranker_accepted:
        # Relevant direct evidence remains useful, but cannot settle every
        # requested facet when coverage/reranking does not support it.
        status = "partial"
    else:
        status = "answerable"
        reasons.append("direct_evidence_covers_requested_information")

    return AnswerabilityDecision(
        status=status,
        reasons=tuple(reasons),
        query_term_coverage=context.query_term_coverage,
        requested_anchors=tuple(sorted(requested)),
        supported_anchors=tuple(sorted(supported)),
        context_sufficiency=context,
    )
