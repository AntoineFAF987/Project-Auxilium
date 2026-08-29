from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from statistics import median
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import unicodedata

import numpy as np

from .constants import EXACT_MATCH_BONUS, EXACT_MATCH_MIN_CHARS, FINAL_K, HYBRID_ALPHA, MAX_CONTEXT_CHARS


@dataclass(frozen=True)
class ScopeRelation:
    name: str
    priority: int
    distance: int


def _zone(meta: Dict[str, Any]) -> Tuple[Any, ...]:
    if meta.get("source") == "email":
        return (meta.get("document_id"), "message")
    return (
        meta.get("document_id"),
        meta.get("section_id") or ("page", meta.get("page")) or "document",
    )


def _relation(anchor: Dict[str, Any], candidate: Dict[str, Any]) -> Optional[ScopeRelation]:
    if anchor.get("chunk_uid") == candidate.get("chunk_uid"):
        return ScopeRelation("anchor", 0, 0)
    anchor_blocks = set(anchor.get("block_ids") or [])
    candidate_blocks = set(candidate.get("block_ids") or [])
    distance = abs(int(anchor.get("order", 0)) - int(candidate.get("order", 0)))
    if anchor_blocks & candidate_blocks:
        return ScopeRelation("same_blocks", 1, distance)

    if anchor.get("source") == "email":
        if anchor.get("document_id") == candidate.get("document_id"):
            return ScopeRelation("same_message", 2, distance)
        anchor_meta = anchor.get("document_metadata") or {}
        candidate_meta = candidate.get("document_metadata") or {}
        thread_id = anchor_meta.get("thread_id")
        if thread_id and thread_id == candidate_meta.get("thread_id"):
            a_key = str(anchor_meta.get("chronological_key") or "")
            c_key = str(candidate_meta.get("chronological_key") or "")
            return ScopeRelation("same_thread", 4, 0 if a_key == c_key else 1)
        attachment_ids = {
            item.get("document_id") for item in anchor_meta.get("attachments", [])
            if isinstance(item, dict)
        }
        if candidate.get("document_id") in attachment_ids:
            return ScopeRelation("linked_attachment", 3, 1)
        return None

    if anchor.get("document_id") != candidate.get("document_id"):
        return None
    if anchor.get("section_id") and anchor.get("section_id") == candidate.get("section_id"):
        return ScopeRelation("same_section", 2, distance)
    if candidate.get("chunk_uid") in {
        anchor.get("previous_chunk_uid"), anchor.get("next_chunk_uid")
    }:
        return ScopeRelation("direct_neighbor", 3, 1)
    anchor_page, candidate_page = anchor.get("page"), candidate.get("page")
    if anchor_page is not None and candidate_page is not None:
        page_distance = abs(int(anchor_page) - int(candidate_page))
        if page_distance == 0:
            return ScopeRelation("same_page", 3, distance)
        if page_distance == 1:
            return ScopeRelation("adjacent_page", 5, page_distance)
    return None


def _score_value(signal: Dict[str, Any]) -> float:
    for name in ("reranker_score", "hybrid_score", "dense_score", "bm25_score"):
        value = signal.get(name)
        if value is not None:
            return float(value)
    return 0.0


def _normalized_query(query: str) -> str:
    normalized = unicodedata.normalize("NFKD", query.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def should_expand_scope(
    *,
    query: str,
    baseline_ids: Sequence[int],
    candidate_signals: Dict[int, Dict[str, Any]],
    metas: Sequence[Dict[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    """Decide whether structural expansion is justified by existing signals.

    The reranker used here returns probabilities, so 0.5 is its natural
    decision boundary rather than a new retrieval tuning parameter. Expansion
    is reserved for explicitly multi-evidence questions whose initial top
    result is not independently convincing.
    """
    if not baseline_ids:
        return False, {"reason": "no_baseline_candidates"}

    normalized = _normalized_query(query)
    interrogative_patterns = (
        r"\bquel(?:le|les|s)?\b", r"\bcombien\b", r"\bpourquoi\b",
        r"\bcomment\b", r"\bquand\b", r"\bou\b", r"\bqui\b",
        r"\bwhat\b", r"\bwhich\b", r"\bwhy\b", r"\bhow\b", r"\bwhen\b",
    )
    interrogative_count = sum(
        len(re.findall(pattern, normalized)) for pattern in interrogative_patterns
    )
    aggregate_terms = (
        "compare", "comparaison", "versus", "retrace", "enumere",
        "tous les", "toutes les", "ainsi que", "all the", "list all",
    )
    matched_terms = [term for term in aggregate_terms if term in normalized]
    multi_evidence_intent = interrogative_count >= 2 or bool(matched_terms)

    scores = [_score_value(candidate_signals.get(idx, {})) for idx in baseline_ids]
    top_score = scores[0]
    top2_score = scores[1] if len(scores) > 1 else None
    top_gap = top_score - top2_score if top2_score is not None else None
    top_exact = bool(
        float(candidate_signals.get(baseline_ids[0], {}).get("exact_match_bonus") or 0.0)
    )
    strong_top = top_score >= 0.5
    top_zones = {_zone(metas[idx]) for idx in baseline_ids[:3]}

    if not multi_evidence_intent:
        expand, reason = False, "single_evidence_intent"
    elif strong_top or top_exact:
        expand, reason = False, "strong_initial_evidence"
    else:
        expand, reason = True, "multi_evidence_without_strong_top"

    return expand, {
        "reason": reason,
        "multi_evidence_intent": multi_evidence_intent,
        "interrogative_count": interrogative_count,
        "matched_aggregate_terms": matched_terms,
        "top_score": top_score,
        "top2_score": top2_score,
        "top_gap": top_gap,
        "top_exact": top_exact,
        "top3_zone_count": len(top_zones),
    }


def select_anchor_scope(
    *,
    query: str,
    query_vec: np.ndarray,
    metas: Sequence[Dict[str, Any]],
    texts: Sequence[str],
    embeddings: np.ndarray,
    bm25_scores: np.ndarray,
    baseline_ids: Sequence[int],
    candidate_signals: Dict[int, Dict[str, Any]],
    cross_encoder: Any = None,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    final_k: int = FINAL_K,
) -> Tuple[List[Tuple[float, int]], Dict[str, Any]]:
    """Select non-redundant anchors, then query-relevant structural scope.

    Anchor and expansion thresholds are distribution-derived (medians of the
    current result set), so this layer adds no new runtime tuning parameter.
    """
    total_started = perf_counter()
    if not baseline_ids:
        return [], {"anchors": [], "scope_candidates": [], "discarded_by_budget": []}

    selection_started = perf_counter()
    baseline_scores = [_score_value(candidate_signals.get(idx, {})) for idx in baseline_ids]
    anchor_floor = float(median(baseline_scores))
    anchor_ids: List[int] = []
    zones = set()
    anchor_trace = []
    for rank, (idx, score) in enumerate(zip(baseline_ids, baseline_scores), start=1):
        meta = metas[idx]
        zone = _zone(meta)
        exact = float(candidate_signals.get(idx, {}).get("exact_match_bonus") or 0.0) > 0
        eligible = rank == 1 or score >= anchor_floor or exact
        reason = (
            "top_reranked" if rank == 1 else
            "exact_lexical" if exact else
            "score_at_or_above_result_median" if score >= anchor_floor else
            "below_result_median"
        )
        selected = eligible and zone not in zones
        if selected:
            zones.add(zone)
            anchor_ids.append(idx)
        anchor_trace.append({
            "candidate_id": idx,
            "chunk_uid": meta.get("chunk_uid"),
            "selected": selected,
            "reason": reason if selected or not eligible else "redundant_zone",
            "score": score,
            "rank": rank,
            "document_id": meta.get("document_id"),
            "section_id": meta.get("section_id"),
            "page": meta.get("page"),
            "zone": list(zone),
        })
    anchor_selection_ms = (perf_counter() - selection_started) * 1000.0

    collection_started = perf_counter()
    relations: Dict[int, List[Tuple[int, ScopeRelation]]] = defaultdict(list)
    scope_ids = set(anchor_ids)
    for anchor_id in anchor_ids:
        for candidate_id, candidate_meta in enumerate(metas):
            relation = _relation(metas[anchor_id], candidate_meta)
            if relation is not None:
                relations[candidate_id].append((anchor_id, relation))
                scope_ids.add(candidate_id)
    structural_collection_ms = (perf_counter() - collection_started) * 1000.0

    scoring_started = perf_counter()
    ordered_scope_ids = sorted(scope_ids)
    dense_raw = np.asarray(
        [float(embeddings[idx] @ query_vec) for idx in ordered_scope_ids], dtype=np.float32
    )
    if dense_raw.size and float(dense_raw.max()) > float(dense_raw.min()):
        dense_norm = (dense_raw - dense_raw.min()) / (dense_raw.max() - dense_raw.min())
    else:
        dense_norm = np.zeros_like(dense_raw)
    query_lower = query.casefold()
    hybrid_scores: Dict[int, float] = {}
    for position, idx in enumerate(ordered_scope_ids):
        exact = (
            EXACT_MATCH_BONUS
            if len(query_lower) >= EXACT_MATCH_MIN_CHARS and query_lower in texts[idx].casefold()
            else 0.0
        )
        hybrid_scores[idx] = float(
            HYBRID_ALPHA * float(bm25_scores[idx])
            + (1.0 - HYBRID_ALPHA) * float(dense_norm[position])
            + exact
        )
    expansion_scoring_ms = (perf_counter() - scoring_started) * 1000.0

    relevance_scores = dict(hybrid_scores)
    reranker_used = False
    reranking_started = perf_counter()
    if cross_encoder is not None and ordered_scope_ids:
        try:
            predicted = cross_encoder.predict([(query, texts[idx]) for idx in ordered_scope_ids])
            relevance_scores = {
                idx: float(score) for idx, score in zip(ordered_scope_ids, predicted)
            }
            reranker_used = True
        except Exception:
            pass
    scope_reranking_ms = (perf_counter() - reranking_started) * 1000.0

    result_relevance = [
        relevance_scores[idx] for idx in baseline_ids if idx in relevance_scores
    ]
    relevance_floor = float(median(result_relevance)) if result_relevance else 0.0
    scope_trace = []
    expansion_ids = []
    anchor_set = set(anchor_ids)
    for idx in ordered_scope_ids:
        if idx in anchor_set:
            continue
        best_anchor, best_relation = min(
            relations[idx], key=lambda item: (item[1].priority, item[1].distance)
        )
        score = relevance_scores[idx]
        same_native_unit = best_relation.name == "same_blocks"
        selected = same_native_unit or score >= relevance_floor
        reason = (
            "same_native_unit" if same_native_unit else
            "relevance_at_or_above_anchor_median" if selected else
            "query_relevance_below_anchor_median"
        )
        if selected:
            expansion_ids.append(idx)
        scope_trace.append({
            "candidate_id": idx,
            "chunk_uid": metas[idx].get("chunk_uid"),
            "anchor_chunk_uid": metas[best_anchor].get("chunk_uid"),
            "relation": best_relation.name,
            "structural_distance": best_relation.distance,
            "relevance_score": score,
            "relevance_floor": relevance_floor,
            "selected": selected,
            "reason": reason,
            "document_id": metas[idx].get("document_id"),
            "section_id": metas[idx].get("section_id"),
            "page": metas[idx].get("page"),
        })

    expansion_ids.sort(key=lambda idx: (
        -relevance_scores[idx],
        min(relation.priority for _, relation in relations[idx]),
        min(relation.distance for _, relation in relations[idx]),
    ))
    priority_ids = list(anchor_ids) + expansion_ids
    selected: List[Tuple[float, int]] = []
    selected_ids = set()
    discarded = []
    used_chars = 0
    for idx in priority_ids:
        if idx in selected_ids:
            continue
        length = len(texts[idx])
        if len(selected) >= final_k or used_chars + length > max_context_chars:
            discarded.append({
                "candidate_id": idx,
                "chunk_uid": metas[idx].get("chunk_uid"),
                "reason": "chunk_count_budget" if len(selected) >= final_k else "character_budget",
                "text_len": length,
            })
            continue
        selected_ids.add(idx)
        used_chars += length
        selected.append((relevance_scores.get(idx, 0.0), idx))

    selected_anchor_count = sum(idx in anchor_set for _, idx in selected)
    selected_scope_count = len(selected) - selected_anchor_count
    return selected, {
        "scope_triggered": True,
        "anchor_floor": anchor_floor,
        "scope_relevance_floor": relevance_floor,
        "reranker_used_for_scope": reranker_used,
        "anchors": anchor_trace,
        "scope_candidates": scope_trace,
        "discarded_by_budget": discarded,
        "selected_anchor_ids": [idx for idx in anchor_ids if idx in selected_ids],
        "selected_scope_ids": [idx for idx in expansion_ids if idx in selected_ids],
        "context_chunk_ids": [idx for _, idx in selected],
        "context_chars": used_chars,
        "context_chunks": len(selected),
        "multi_anchor": selected_anchor_count > 1,
        "multi_chunk_scope": selected_scope_count > 0,
        "rescored_candidates": len(ordered_scope_ids) if reranker_used else 0,
        "timings_ms": {
            "anchor_selection": anchor_selection_ms,
            "structural_collection": structural_collection_ms,
            "expansion_scoring": expansion_scoring_ms,
            "scope_reranking": scope_reranking_ms,
            "total": (perf_counter() - total_started) * 1000.0,
        },
    }


def select_anchor_scope_adaptive(
    *,
    query: str,
    query_vec: np.ndarray,
    metas: Sequence[Dict[str, Any]],
    texts: Sequence[str],
    embeddings: np.ndarray,
    bm25_scores: np.ndarray,
    baseline_ids: Sequence[int],
    candidate_signals: Dict[int, Dict[str, Any]],
    cross_encoder: Any = None,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    final_k: int = FINAL_K,
) -> Tuple[List[Tuple[float, int]], Dict[str, Any]]:
    """Apply a minimal structural correction only when the baseline needs it."""
    total_started = perf_counter()
    decision_started = perf_counter()
    expand, decision = should_expand_scope(
        query=query,
        baseline_ids=baseline_ids,
        candidate_signals=candidate_signals,
        metas=metas,
    )
    decision_ms = (perf_counter() - decision_started) * 1000.0

    def baseline_results() -> List[Tuple[float, int]]:
        return [(_score_value(candidate_signals.get(idx, {})), idx) for idx in baseline_ids]

    if not expand:
        return baseline_results(), {
            "scope_triggered": False,
            "decision": decision,
            "anchors": [],
            "scope_candidates": [],
            "discarded_by_budget": [],
            "selected_anchor_ids": [],
            "selected_scope_ids": [],
            "context_chunk_ids": list(baseline_ids),
            "context_chars": sum(len(texts[idx]) for idx in baseline_ids),
            "context_chunks": len(baseline_ids),
            "multi_anchor": False,
            "multi_chunk_scope": False,
            "rescored_candidates": 0,
            "timings_ms": {
                "anchor_selection": decision_ms,
                "structural_collection": 0.0,
                "expansion_scoring": 0.0,
                "scope_reranking": 0.0,
                "total": (perf_counter() - total_started) * 1000.0,
            },
        }

    selection_started = perf_counter()
    anchor_ids = [baseline_ids[0]]
    comparison_intent = any(
        term in _normalized_query(query)
        for term in ("compare", "comparaison", "versus")
    )
    if comparison_intent:
        first_zone = _zone(metas[anchor_ids[0]])
        second_anchor = next(
            (idx for idx in baseline_ids[1:] if _zone(metas[idx]) != first_zone),
            None,
        )
        if second_anchor is not None:
            anchor_ids.append(second_anchor)
    anchor_selection_ms = (perf_counter() - selection_started) * 1000.0

    anchor_set = set(anchor_ids)
    anchor_trace = []
    for rank, idx in enumerate(baseline_ids, start=1):
        selected_anchor = idx in anchor_set
        anchor_trace.append({
            "candidate_id": idx,
            "chunk_uid": metas[idx].get("chunk_uid"),
            "selected": selected_anchor,
            "reason": (
                "top_reranked" if idx == anchor_ids[0]
                else "explicit_comparison_second_zone" if selected_anchor
                else "adaptive_anchor_not_needed"
            ),
            "score": _score_value(candidate_signals.get(idx, {})),
            "rank": rank,
            "document_id": metas[idx].get("document_id"),
            "section_id": metas[idx].get("section_id"),
            "page": metas[idx].get("page"),
            "zone": list(_zone(metas[idx])),
        })

    collection_started = perf_counter()
    relations: Dict[int, List[Tuple[int, ScopeRelation]]] = defaultdict(list)
    scope_ids = set(anchor_ids)
    for anchor_id in anchor_ids:
        for candidate_id, candidate_meta in enumerate(metas):
            relation = _relation(metas[anchor_id], candidate_meta)
            if relation is not None:
                relations[candidate_id].append((anchor_id, relation))
                scope_ids.add(candidate_id)
    structural_collection_ms = (perf_counter() - collection_started) * 1000.0

    scoring_started = perf_counter()
    ordered_scope_ids = sorted(scope_ids)
    dense_raw = np.asarray(
        [float(embeddings[idx] @ query_vec) for idx in ordered_scope_ids], dtype=np.float32
    )
    if dense_raw.size and float(dense_raw.max()) > float(dense_raw.min()):
        dense_norm = (dense_raw - dense_raw.min()) / (dense_raw.max() - dense_raw.min())
    else:
        dense_norm = np.zeros_like(dense_raw)
    query_lower = query.casefold()
    cheap_scores: Dict[int, float] = {}
    for position, idx in enumerate(ordered_scope_ids):
        exact = (
            EXACT_MATCH_BONUS
            if len(query_lower) >= EXACT_MATCH_MIN_CHARS and query_lower in texts[idx].casefold()
            else 0.0
        )
        cheap_scores[idx] = float(
            HYBRID_ALPHA * float(bm25_scores[idx])
            + (1.0 - HYBRID_ALPHA) * float(dense_norm[position])
            + exact
        )

    # Minimal breadth: keep only the best cheap candidate for each structural
    # relation of each anchor before invoking the expensive cross-encoder.
    best_by_relation: Dict[Tuple[int, str], int] = {}
    for idx, linked_anchors in relations.items():
        if idx in anchor_set:
            continue
        for anchor_id, relation in linked_anchors:
            key = (anchor_id, relation.name)
            current = best_by_relation.get(key)
            if current is None or (
                cheap_scores[idx], -relation.distance
            ) > (
                cheap_scores[current],
                -min(rel.distance for aid, rel in relations[current] if aid == anchor_id),
            ):
                best_by_relation[key] = idx
    shortlist_ids = list(dict.fromkeys(best_by_relation.values()))
    expansion_scoring_ms = (perf_counter() - scoring_started) * 1000.0

    relevance_scores = dict(cheap_scores)
    to_rescore = []
    for idx in shortlist_ids:
        existing = candidate_signals.get(idx, {}).get("reranker_score")
        if existing is not None:
            relevance_scores[idx] = float(existing)
        elif cross_encoder is not None:
            to_rescore.append(idx)

    reranking_started = perf_counter()
    reranker_used = False
    if to_rescore:
        try:
            predicted = cross_encoder.predict([(query, texts[idx]) for idx in to_rescore])
            for idx, score in zip(to_rescore, predicted):
                relevance_scores[idx] = float(score)
            reranker_used = True
        except Exception:
            to_rescore = []
    scope_reranking_ms = (perf_counter() - reranking_started) * 1000.0

    baseline_scores = [_score_value(candidate_signals.get(idx, {})) for idx in baseline_ids]
    relevance_floor = float(median(baseline_scores))
    scope_trace = []
    expansion_ids = []
    for idx in shortlist_ids:
        best_anchor, best_relation = min(
            relations[idx], key=lambda item: (item[1].priority, item[1].distance)
        )
        score = relevance_scores[idx]
        selected_expansion = score >= relevance_floor
        if selected_expansion:
            expansion_ids.append(idx)
        scope_trace.append({
            "candidate_id": idx,
            "chunk_uid": metas[idx].get("chunk_uid"),
            "anchor_chunk_uid": metas[best_anchor].get("chunk_uid"),
            "relation": best_relation.name,
            "structural_distance": best_relation.distance,
            "relevance_score": score,
            "relevance_floor": relevance_floor,
            "selected": selected_expansion,
            "reason": (
                "relevance_at_or_above_baseline_median"
                if selected_expansion else "query_relevance_below_baseline_median"
            ),
            "document_id": metas[idx].get("document_id"),
            "section_id": metas[idx].get("section_id"),
            "page": metas[idx].get("page"),
        })

    if not expansion_ids:
        selected = baseline_results()
        selected_ids = set(baseline_ids)
        discarded = []
        used_chars = sum(len(texts[idx]) for idx in baseline_ids)
    else:
        expansion_ids.sort(key=lambda idx: (
            -relevance_scores[idx],
            min(relation.priority for _, relation in relations[idx]),
            min(relation.distance for _, relation in relations[idx]),
        ))
        priority_ids = list(anchor_ids) + expansion_ids + list(baseline_ids)
        selected = []
        selected_ids = set()
        discarded = []
        used_chars = 0
        for idx in priority_ids:
            if idx in selected_ids:
                continue
            length = len(texts[idx])
            if len(selected) >= final_k or used_chars + length > max_context_chars:
                discarded.append({
                    "candidate_id": idx,
                    "chunk_uid": metas[idx].get("chunk_uid"),
                    "reason": "chunk_count_budget" if len(selected) >= final_k else "character_budget",
                    "text_len": length,
                })
                continue
            selected_ids.add(idx)
            used_chars += length
            selected.append((relevance_scores.get(
                idx, _score_value(candidate_signals.get(idx, {}))
            ), idx))

    selected_anchor_count = sum(idx in anchor_set for _, idx in selected)
    selected_scope_ids = [idx for idx in expansion_ids if idx in selected_ids]
    return selected, {
        "scope_triggered": True,
        "decision": decision,
        "scope_relevance_floor": relevance_floor,
        "reranker_used_for_scope": reranker_used,
        "anchors": anchor_trace,
        "scope_candidates": scope_trace,
        "discarded_by_budget": discarded,
        "selected_anchor_ids": [idx for idx in anchor_ids if idx in selected_ids],
        "selected_scope_ids": selected_scope_ids,
        "context_chunk_ids": [idx for _, idx in selected],
        "context_chars": used_chars,
        "context_chunks": len(selected),
        "multi_anchor": selected_anchor_count > 1,
        "multi_chunk_scope": bool(selected_scope_ids),
        "structural_candidates": len(scope_ids) - len(anchor_ids),
        "shortlisted_candidates": len(shortlist_ids),
        "rescored_candidates": len(to_rescore),
        "timings_ms": {
            "anchor_selection": decision_ms + anchor_selection_ms,
            "structural_collection": structural_collection_ms,
            "expansion_scoring": expansion_scoring_ms,
            "scope_reranking": scope_reranking_ms,
            "total": (perf_counter() - total_started) * 1000.0,
        },
    }
