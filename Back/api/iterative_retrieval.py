"""Bounded, deterministic evidence enrichment around the existing RAG search.

This module never queries FAISS/BM25, generates text, or invents identifiers.
It only follows relations already present in retrieved metadata and the loaded
corpus.  It is intentionally a finite state machine, not an agent loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal


ActionType = Literal["SEARCH", "EXPAND", "FOLLOW", "ANSWER"]
QuerySemantics = Literal[
    "fact_lookup", "current_state", "decision", "chronology", "comparison",
    "procedure", "general_document_question",
]


@dataclass(frozen=True)
class RetrievalAction:
    type: ActionType
    target_document_id: str | None = None
    relation: Literal["adjacent_chunks", "attachment", "same_thread"] | None = None
    query: str | None = None


@dataclass(frozen=True)
class EvidenceSufficiency:
    sufficient: bool
    reason: str
    next_action: RetrievalAction | None = None


@dataclass
class EvidenceSet:
    """Cumulative, de-duplicated evidence across a maximum of three rounds."""
    items: list[tuple[float, dict[str, Any]]] = field(default_factory=list)
    seen_chunk_uids: set[str] = field(default_factory=set)
    seen_document_ids: set[str] = field(default_factory=set)
    followed_attachments: set[str] = field(default_factory=set)
    followed_threads: set[str] = field(default_factory=set)
    expanded_documents: set[str] = field(default_factory=set)

    def add(self, rows: list[tuple[float, dict[str, Any]]]) -> list[tuple[float, dict[str, Any]]]:
        added: list[tuple[float, dict[str, Any]]] = []
        for score, meta in rows:
            uid = str(meta.get("chunk_uid") or f"{meta.get('document_id')}:{meta.get('chunk_id')}")
            fused_uids = {str(item) for item in (meta.get("fused_chunk_uids") or []) if item}
            all_uids = fused_uids or {uid}
            if uid in self.seen_chunk_uids:
                continue
            self.seen_chunk_uids.update(all_uids)
            if meta.get("document_id"):
                self.seen_document_ids.add(str(meta["document_id"]))
            item = (float(score), dict(meta))
            self.items.append(item)
            added.append(item)
        return added


@dataclass(frozen=True)
class CandidatePool:
    """Search candidates retained before final-context clipping."""
    items: tuple[tuple[float, dict[str, Any]], ...]

    def by_document_id(self, document_id: str) -> dict[str, Any] | None:
        return next((meta for _, meta in self.items if str(meta.get("document_id")) == document_id), None)

    def trace_items(self, limit: int = 20) -> list[dict[str, Any]]:
        return [{
            "document_id": meta.get("document_id"), "chunk_uid": meta.get("chunk_uid"),
            "score": score, "source": meta.get("source"), "date": _date_key(meta),
            "title": meta.get("title") or (meta.get("document_metadata") or {}).get("subject"),
        } for score, meta in self.items[:limit]]


@dataclass(frozen=True)
class EvidenceGap:
    """Actionable content missing from a document already used as evidence."""
    document_id: str
    chunk_uid: str | None
    gap_type: Literal["partial_document", "email_header_only", "unfollowed_attachment", "adjacent_content_available"]
    available_actions: tuple[RetrievalAction, ...]
    date: str
    relevance: float
    reason: str
    semantic_fit: Literal["direct_subject_match", "state_resolution_context", "weak_selected_evidence"]
    temporal_relation: Literal["newer_than_existing_state_evidence", "recent_relative_to_selected_evidence", "not_newer", "undated"]
    state_change_potential: bool
    priority_class: Literal["potential_newer_state_resolution", "recent_directly_related", "temporal_coverage", "selected_evidence_expansion", "retrieval_relevance"]
    priority_reason: str


@dataclass(frozen=True)
class CandidateLead:
    """Actionable, related search candidate not yet loaded as evidence."""
    document_id: str
    chunk_uid: str | None
    score: float
    date: str
    source: str | None
    available_actions: tuple[RetrievalAction, ...]
    reason: str


def _date_key(meta: dict[str, Any]) -> str:
    data = meta.get("document_metadata") or {}
    return str(data.get("chronological_key") or data.get("date") or "")


def _is_header_only(meta: dict[str, Any]) -> bool:
    blocks = meta.get("blocks") or []
    types = {str(block.get("block_type")) for block in blocks if isinstance(block, dict)}
    if types:
        if "email_header" not in types or "email_body" in types:
            return False
        # Fused neighbouring chunks can contain a body while retaining the
        # anchor chunk's block metadata.
        return "\n\n" not in str(meta.get("text") or "")
    text = str(meta.get("text") or "").lower()
    return meta.get("source") == "email" and text.startswith(("subject:", "objet:"))


def _rows_for_document(corpus: list[dict[str, Any]], document_id: str) -> list[dict[str, Any]]:
    return [row for row in corpus if str(row.get("document_id")) == document_id]


def _attachment_ids(meta: dict[str, Any]) -> list[str]:
    metadata = meta.get("document_metadata") or {}
    return [
        str(item.get("document_id")) for item in metadata.get("attachments", [])
        if isinstance(item, dict) and item.get("relation_type") == "attachment" and item.get("document_id")
    ]


def _thread_id(meta: dict[str, Any]) -> str | None:
    value = (meta.get("document_metadata") or {}).get("thread_id")
    return str(value) if value else None


def _terms(value: str) -> set[str]:
    """Terms useful for topical matching, excluding question scaffolding."""
    stopwords = {
        "the", "was", "are", "and", "for", "with", "what", "was", "my", "your",
        "quel", "quelle", "dans", "pour", "avec", "est", "une", "des", "les", "que",
        "request", "demande", "final", "finally", "decision", "décision", "status", "statut",
        "subject", "objet", "regarding", "concerning", "about", "from", "sent",
    }
    return {term for term in re.findall(r"[a-z0-9à-ÿ]{3,}", value.lower()) if term not in stopwords}


def _raw_terms(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9à-ÿ]{3,}", value.lower()))


def _proper_terms(value: str) -> set[str]:
    """Conservative heuristic for names/labels; they cannot alone prove topic."""
    return {
        re.sub(r"['’]s$", "", term.lower())
        for term in re.findall(r"\b[A-ZÀ-Ö][A-Za-zÀ-ÿ'-]{2,}\b", value)
    }


_STATE_CHANGE_TERMS = {
    "approved", "rejected", "accepted", "refused", "pending", "validated", "cancelled", "closed",
    "completed", "signed", "updated", "update", "changed", "change", "final", "status", "decision",
    "accepté", "refusé", "validé", "annulé", "clôturé", "terminé", "signé", "modifié",
    "statut", "décision", "final",
}


def _title_text(meta: dict[str, Any]) -> str:
    metadata = meta.get("document_metadata") or {}
    return " ".join((str(meta.get("title") or ""), str(metadata.get("subject") or ""), str(meta.get("text") or "")[:400]))


def _topic_relation(meta: dict[str, Any], query: str, semantics: QuerySemantics) -> tuple[Literal["direct_subject_match", "state_resolution_context", "weak_selected_evidence"], bool]:
    """Return semantic fit and whether the title supplies a state-change lead.

    Shared names, organisations and location labels do not establish topical
    relation by themselves.  They need an accompanying non-entity topic term.
    """
    title = _title_text(meta)
    query_terms, title_terms = _terms(query), _terms(title)
    shared = query_terms & title_terms
    proper_shared = _proper_terms(query) & _proper_terms(title)
    non_entity_shared = shared - proper_shared
    state_potential = bool(_STATE_CHANGE_TERMS & _raw_terms(title))
    title_topics = title_terms - _STATE_CHANGE_TERMS - proper_shared
    query_topics = query_terms - _STATE_CHANGE_TERMS - proper_shared
    competing_topic = bool(title_topics - query_topics) and not non_entity_shared
    if non_entity_shared:
        return "direct_subject_match", state_potential
    if semantics in {"current_state", "decision"} and state_potential and not competing_topic:
        return "state_resolution_context", True
    return "weak_selected_evidence", state_potential


def _candidate_is_related(meta: dict[str, Any], query: str, semantics: QuerySemantics) -> bool:
    """Require topical relation; CandidatePool membership alone is not evidence."""
    fit, _state_potential = _topic_relation(meta, query, semantics)
    return fit in {"direct_subject_match", "state_resolution_context"}


def _available_lead_action(meta: dict[str, Any], evidence: EvidenceSet, corpus: list[dict[str, Any]]) -> RetrievalAction | None:
    document_id = str(meta.get("document_id") or "")
    if not document_id:
        return None
    for attachment_id in _attachment_ids(meta):
        if attachment_id not in evidence.followed_attachments and _rows_for_document(corpus, attachment_id):
            return RetrievalAction("FOLLOW", document_id, "attachment")
    if document_id not in evidence.expanded_documents and any(str(row.get("chunk_uid")) not in evidence.seen_chunk_uids for row in _rows_for_document(corpus, document_id)):
        return RetrievalAction("EXPAND", document_id, "adjacent_chunks")
    thread_id = _thread_id(meta)
    if thread_id and thread_id not in evidence.followed_threads:
        if any(str((row.get("document_metadata") or {}).get("thread_id") or "") == thread_id and str(row.get("chunk_uid")) not in evidence.seen_chunk_uids for row in corpus):
            return RetrievalAction("FOLLOW", document_id, "same_thread")
    return None


def _gap_priority(
    *, meta: dict[str, Any], score: float, evidence: EvidenceSet, query: str, semantics: QuerySemantics,
) -> tuple[
    Literal["direct_subject_match", "state_resolution_context", "weak_selected_evidence"],
    Literal["newer_than_existing_state_evidence", "recent_relative_to_selected_evidence", "not_newer", "undated"],
    bool,
    Literal["potential_newer_state_resolution", "recent_directly_related", "temporal_coverage", "selected_evidence_expansion", "retrieval_relevance"],
    str,
    tuple[int, str, float],
]:
    """Classify an EvidenceGap without modifying retrieval scores or ranking.

    The returned tuple is a lexicographic execution priority.  Date is used
    only after semantic class selection; it is never a standalone boost.
    """
    semantic_fit, state_change_potential = _topic_relation(meta, query, semantics)
    date = _date_key(meta)
    other_dates = [_date_key(item) for _, item in evidence.items if _date_key(item) and _date_key(item) != date]
    # Supersession requires being newer than the currently observed evidence,
    # not merely newer than one old item while a later item already exists.
    newer = bool(date and other_dates and date > max(other_dates))
    if not date:
        temporal_relation: Literal["newer_than_existing_state_evidence", "recent_relative_to_selected_evidence", "not_newer", "undated"] = "undated"
    elif newer and semantics in {"current_state", "decision"} and state_change_potential:
        temporal_relation = "newer_than_existing_state_evidence"
    elif newer:
        temporal_relation = "recent_relative_to_selected_evidence"
    else:
        temporal_relation = "not_newer"

    if semantics in {"current_state", "decision"} and semantic_fit != "weak_selected_evidence" and newer and state_change_potential:
        priority_class: Literal["potential_newer_state_resolution", "recent_directly_related", "temporal_coverage", "selected_evidence_expansion", "retrieval_relevance"] = "potential_newer_state_resolution"
        priority_reason = "newer relevant partial evidence may supersede older state evidence"
        execution_key = (0, "", -score)
    elif semantics in {"current_state", "decision"} and semantic_fit == "direct_subject_match" and newer:
        priority_class = "recent_directly_related"
        priority_reason = "newer directly related evidence should be inspected before older supporting evidence"
        execution_key = (1, "", -score)
    elif semantics == "chronology" and date and date not in other_dates:
        priority_class = "temporal_coverage"
        priority_reason = "uninspected evidence adds a distinct point to the requested chronology"
        execution_key = (2, date, -score)
    elif semantic_fit != "weak_selected_evidence":
        priority_class = "selected_evidence_expansion"
        priority_reason = "selected relevant evidence has available content or a structured relation"
        execution_key = (3, "", -score)
    else:
        priority_class = "retrieval_relevance"
        priority_reason = "selected evidence has an available relation but weak semantic support"
        execution_key = (4, "", -score)
    # Date resolves ties inside a semantic class for state questions; retrieval
    # relevance remains the final deterministic tie breaker.
    # Recency only resolves ties in classes where recency is already semantic.
    date_key = ""
    if priority_class in {"potential_newer_state_resolution", "recent_directly_related", "temporal_coverage"}:
        date_key = "".join(chr(0x10FFFF - ord(ch)) for ch in date)
    return semantic_fit, temporal_relation, state_change_potential, priority_class, priority_reason, (execution_key[0], date_key, execution_key[2])


def _evidence_gaps(
    evidence: EvidenceSet, *, corpus: list[dict[str, Any]], query: str, semantics: QuerySemantics,
) -> list[EvidenceGap]:
    """Find uninspected relations on documents that already support evidence.

    A loaded header marks a chunk as seen, not its document as complete.
    """
    decorated_gaps: list[tuple[tuple[int, str, float], EvidenceGap]] = []
    seen_documents: set[str] = set()
    for score, meta in evidence.items:
        document_id = str(meta.get("document_id") or "")
        if not document_id or document_id in seen_documents:
            continue
        seen_documents.add(document_id)
        # Evidence has already passed the normal selection. Its actionable
        # structured relation is therefore examined before speculative pool
        # candidates; loading one chunk never means its document is complete.
        unseen_rows = [row for row in _rows_for_document(corpus, document_id) if str(row.get("chunk_uid")) not in evidence.seen_chunk_uids]
        actions: list[RetrievalAction] = []
        # Complete the selected document before following outward relations.
        if unseen_rows and document_id not in evidence.expanded_documents:
            actions.append(RetrievalAction("EXPAND", document_id, "adjacent_chunks"))
        for attachment_id in _attachment_ids(meta):
            if attachment_id not in evidence.followed_attachments and _rows_for_document(corpus, attachment_id):
                actions.append(RetrievalAction("FOLLOW", document_id, "attachment"))
                break
        thread_id = _thread_id(meta)
        if thread_id and thread_id not in evidence.followed_threads:
            if any(str((row.get("document_metadata") or {}).get("thread_id") or "") == thread_id and str(row.get("chunk_uid")) not in evidence.seen_chunk_uids for row in corpus):
                actions.append(RetrievalAction("FOLLOW", document_id, "same_thread"))
        if not actions:
            continue
        if _is_header_only(meta):
            gap_type: Literal["partial_document", "email_header_only", "unfollowed_attachment", "adjacent_content_available"] = "email_header_only"
        elif any(action.relation == "attachment" for action in actions):
            gap_type = "unfollowed_attachment"
        else:
            gap_type = "adjacent_content_available"
        semantic_fit, temporal_relation, state_change_potential, priority_class, priority_reason, execution_key = _gap_priority(
            meta=meta, score=float(score), evidence=evidence, query=query, semantics=semantics,
        )
        decorated_gaps.append((execution_key, EvidenceGap(
            document_id=document_id, chunk_uid=meta.get("chunk_uid"), gap_type=gap_type,
            available_actions=tuple(actions), date=_date_key(meta), relevance=float(score),
            reason="selected evidence is partial and has an uninspected structured relation",
            semantic_fit=semantic_fit, temporal_relation=temporal_relation,
            state_change_potential=state_change_potential, priority_class=priority_class,
            priority_reason=priority_reason,
        )))
    return [gap for _key, gap in sorted(decorated_gaps, key=lambda item: item[0])]


def _candidate_leads(
    pool: CandidatePool, evidence: EvidenceSet, *, corpus: list[dict[str, Any]], query: str, semantics: QuerySemantics,
) -> list[CandidateLead]:
    leads: list[CandidateLead] = []
    for score, meta in pool.items:
        uid = str(meta.get("chunk_uid") or "")
        if uid in evidence.seen_chunk_uids or not _candidate_is_related(meta, query, semantics):
            continue
        action = _available_lead_action(meta, evidence, corpus)
        if action:
            leads.append(CandidateLead(
                document_id=str(meta.get("document_id") or ""), chunk_uid=meta.get("chunk_uid"), score=float(score),
                date=_date_key(meta), source=meta.get("source"), available_actions=(action,),
                reason="topically_related_uninspected_structured_candidate",
            ))
    return leads


def _priority_items(evidence: EvidenceSet, semantics: QuerySemantics) -> list[tuple[float, dict[str, Any]]]:
    if semantics not in {"current_state", "decision"}:
        return list(evidence.items)
    # State/decision queries favour later structured evidence. This changes
    # only final evidence ordering, not the existing retrieval ranking.
    return sorted(evidence.items, key=lambda item: (_date_key(item[1]), item[0]), reverse=True)


def _has_answer_bearing_content(items: list[tuple[float, dict[str, Any]]], semantics: QuerySemantics) -> bool:
    if semantics in {"fact_lookup", "general_document_question"}:
        return any(not _is_header_only(meta) for _, meta in items)
    if semantics == "chronology":
        return len({_date_key(meta)[:10] for _, meta in items if _date_key(meta)}) >= 2
    # A mention of a decision alone (for example, "see attached decision") is
    # not evidence of its outcome.  Require an actual state/result marker.
    state_markers = ("approved", "rejected", "pending", "cancelled", "closed", "validated", "accepted", "refused", "en attente", "accepte", "refuse", "valide", "annule", "cloture")
    return any(not _is_header_only(meta) and any(marker in str(meta.get("text") or "").lower() for marker in state_markers) for _, meta in items)


def _inspect(
    evidence: EvidenceSet, pool: CandidatePool, *, corpus: list[dict[str, Any]], query: str, semantics: QuerySemantics,
) -> EvidenceSufficiency:
    items = _priority_items(evidence, semantics)
    if not items:
        return EvidenceSufficiency(False, "no_retrieved_evidence")
    gaps = _evidence_gaps(evidence, corpus=corpus, query=query, semantics=semantics)
    leads = _candidate_leads(pool, evidence, corpus=corpus, query=query, semantics=semantics)
    # First complete selected partial evidence. A header is a lead, not the
    # document's answer-bearing content.
    expansion_gaps = [gap for gap in gaps if gap.available_actions and gap.available_actions[0].type == "EXPAND"]
    answer_available = _has_answer_bearing_content(items, semantics)
    if expansion_gaps and semantics in {"decision", "current_state", "chronology"} and (
        not answer_available
        or expansion_gaps[0].priority_class in {"potential_newer_state_resolution", "recent_directly_related", "temporal_coverage"}
    ):
        return EvidenceSufficiency(False, "semantic_question_has_unresolved_selected_evidence_gap", expansion_gaps[0].available_actions[0])
    if leads and semantics in {"decision", "current_state", "chronology"}:
        return EvidenceSufficiency(False, "semantic_question_has_unresolved_answer_bearing_candidate", leads[0].available_actions[0])
    if answer_available:
        return EvidenceSufficiency(True, "semantic_answer_bearing_evidence_available")
    if gaps and semantics in {"decision", "current_state", "chronology"}:
        return EvidenceSufficiency(False, "semantic_question_has_unresolved_selected_evidence_gap", gaps[0].available_actions[0])
    if any(_is_header_only(meta) for _, meta in items):
        return EvidenceSufficiency(False, "relevant_document_is_partial_without_new_relation")
    return EvidenceSufficiency(False, "evidence_is_not_answer_bearing_for_query_semantics")


def _execute_action(
    action: RetrievalAction, evidence: EvidenceSet, pool: CandidatePool, corpus: list[dict[str, Any]], *, max_expanded_chunks: int,
) -> list[tuple[float, dict[str, Any]]]:
    if action.type == "EXPAND" and action.target_document_id and action.target_document_id not in evidence.expanded_documents:
        evidence.expanded_documents.add(action.target_document_id)
        rows = [row for row in _rows_for_document(corpus, action.target_document_id) if str(row.get("chunk_uid")) not in evidence.seen_chunk_uids]
        rows.sort(key=lambda row: int(row.get("order") or 0))
        return [(0.0, row) for row in rows[:max_expanded_chunks]]
    if action.type == "FOLLOW" and action.relation == "attachment" and action.target_document_id:
        origin = pool.by_document_id(action.target_document_id) or next((meta for _, meta in evidence.items if str(meta.get("document_id")) == action.target_document_id), None)
        if not origin:
            return []
        attachment_ids = [item for item in _attachment_ids(origin) if item not in evidence.followed_attachments]
        if not attachment_ids:
            return []
        attachment_id = attachment_ids[0]
        evidence.followed_attachments.add(attachment_id)
        return [(0.0, row) for row in _rows_for_document(corpus, attachment_id) if str(row.get("chunk_uid")) not in evidence.seen_chunk_uids][:max_expanded_chunks]
    if action.type == "FOLLOW" and action.relation == "same_thread" and action.target_document_id:
        origin = pool.by_document_id(action.target_document_id) or next((meta for _, meta in evidence.items if str(meta.get("document_id")) == action.target_document_id), None)
        thread_id = _thread_id(origin or {})
        if not thread_id or thread_id in evidence.followed_threads:
            return []
        evidence.followed_threads.add(thread_id)
        return [
            (0.0, row) for row in corpus
            if str((row.get("document_metadata") or {}).get("thread_id") or "") == thread_id
            and str(row.get("chunk_uid")) not in evidence.seen_chunk_uids
        ][:max_expanded_chunks]
    return []


def _trace_gaps(gaps: list[EvidenceGap]) -> list[dict[str, Any]]:
    return [{
        "document_id": gap.document_id, "chunk_uid": gap.chunk_uid, "gap_type": gap.gap_type,
        "available_actions": [action.type for action in gap.available_actions], "date": gap.date,
        "relevance": gap.relevance, "reason": gap.reason,
        "semantic_fit": gap.semantic_fit, "temporal_relation": gap.temporal_relation,
        "state_change_potential": gap.state_change_potential,
        "priority_class": gap.priority_class, "priority_reason": gap.priority_reason,
    } for gap in gaps]


def _trace_leads(leads: list[CandidateLead]) -> list[dict[str, Any]]:
    return [{
        "document_id": lead.document_id, "chunk_uid": lead.chunk_uid, "score": lead.score,
        "date": lead.date, "source": lead.source,
        "available_actions": [action.type for action in lead.available_actions], "reason": lead.reason,
    } for lead in leads]


def _lead_trace_fields(
    action: RetrievalAction | None, gaps: list[EvidenceGap], leads: list[CandidateLead],
) -> dict[str, Any]:
    if action and any(action == gap.available_actions[0] for gap in gaps):
        return {
            "selected_lead_type": "evidence_gap", "selected_lead": action.target_document_id,
            "lead_priority_reason": "complete already-selected relevant evidence before exploring new candidates",
        }
    if action and any(action == lead.available_actions[0] for lead in leads):
        return {
            "selected_lead_type": "candidate_lead", "selected_lead": action.target_document_id,
            "lead_priority_reason": "inspect a topically related unselected candidate with an available structured action",
        }
    return {"selected_lead_type": None, "selected_lead": None, "lead_priority_reason": None}


def run_iterative_evidence_retrieval(
    *, candidate_pool: list[tuple[float, dict[str, Any]]], initial_evidence: list[tuple[float, dict[str, Any]]], corpus: list[dict[str, Any]],
    query: str, semantics: QuerySemantics = "general_document_question", max_rounds: int = 3,
    max_expanded_chunks: int = 4,
) -> tuple[list[tuple[float, dict[str, Any]]], list[dict[str, Any]], EvidenceSufficiency]:
    """Enrich evidence using only valid, unseen, structured relations."""
    pool = CandidatePool(tuple(candidate_pool))
    evidence = EvidenceSet()
    initial_added = evidence.add(initial_evidence)
    rounds: list[dict[str, Any]] = [{
        "round": 1, "action": {"type": "SEARCH", "query": query},
        "candidate_pool_count": len(pool.items), "candidate_pool": pool.trace_items(),
        "evidence_count": len(evidence.items),
        "added_chunk_uids": [meta.get("chunk_uid") for _, meta in initial_added],
    }]
    decision = _inspect(evidence, pool, corpus=corpus, query=query, semantics=semantics)
    gaps = _evidence_gaps(evidence, corpus=corpus, query=query, semantics=semantics)
    leads = _candidate_leads(pool, evidence, corpus=corpus, query=query, semantics=semantics)
    rounds[-1]["evidence_gaps"] = _trace_gaps(gaps)
    rounds[-1]["candidate_leads"] = _trace_leads(leads)
    # Kept for trace compatibility; it now means unselected CandidateLeads.
    rounds[-1]["unresolved_leads"] = rounds[-1]["candidate_leads"]
    rounds[-1].update(_lead_trace_fields(decision.next_action, gaps, leads))
    rounds[-1]["sufficiency"] = {
        "sufficient": decision.sufficient, "reason": decision.reason,
        "next_action": decision.next_action.__dict__ if decision.next_action else None,
    }

    while not decision.sufficient and len(rounds) < max_rounds and decision.next_action:
        action = decision.next_action
        selected_fields = _lead_trace_fields(action, gaps, leads)
        added = evidence.add(_execute_action(action, evidence, pool, corpus, max_expanded_chunks=max_expanded_chunks))
        round_data = {
            "round": len(rounds) + 1, "action": action.__dict__,
            "candidate_pool_count": len(pool.items), "evidence_count": len(evidence.items),
            "added_chunk_uids": [meta.get("chunk_uid") for _, meta in added],
        }
        if not added:
            decision = EvidenceSufficiency(False, "requested_action_added_no_new_evidence")
        else:
            decision = _inspect(evidence, pool, corpus=corpus, query=query, semantics=semantics)
        gaps = _evidence_gaps(evidence, corpus=corpus, query=query, semantics=semantics)
        leads = _candidate_leads(pool, evidence, corpus=corpus, query=query, semantics=semantics)
        round_data["evidence_gaps"] = _trace_gaps(gaps)
        round_data["candidate_leads"] = _trace_leads(leads)
        round_data["unresolved_leads"] = round_data["candidate_leads"]
        round_data.update(selected_fields)
        round_data["sufficiency"] = {
            "sufficient": decision.sufficient, "reason": decision.reason,
            "next_action": decision.next_action.__dict__ if decision.next_action else None,
        }
        rounds.append(round_data)

    if not decision.sufficient and len(rounds) >= max_rounds:
        decision = EvidenceSufficiency(False, "max_rounds_reached")
        rounds[-1]["sufficiency"] = {"sufficient": False, "reason": decision.reason, "next_action": None}
    return _priority_items(evidence, semantics), rounds, decision
