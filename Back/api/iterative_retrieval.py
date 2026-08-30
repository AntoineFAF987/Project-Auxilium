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
    return {term for term in re.findall(r"[a-z0-9]{3,}", value.lower()) if term not in {"the", "was", "are", "and", "for", "with", "what", "quel", "quelle", "dans", "pour", "avec"}}


def _candidate_is_related(meta: dict[str, Any], query: str) -> bool:
    """A candidate already came from search; require a lexical lead as well."""
    title = " ".join((str(meta.get("title") or ""), str((meta.get("document_metadata") or {}).get("subject") or ""), str(meta.get("text") or "")[:400]))
    query_terms, title_terms = _terms(query), _terms(title)
    return bool(query_terms & title_terms)


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


def _unresolved_leads(pool: CandidatePool, evidence: EvidenceSet, *, corpus: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    leads = []
    for score, meta in pool.items:
        uid = str(meta.get("chunk_uid") or "")
        if uid in evidence.seen_chunk_uids or not _candidate_is_related(meta, query):
            continue
        action = _available_lead_action(meta, evidence, corpus)
        if action:
            leads.append({
                "document_id": meta.get("document_id"), "chunk_uid": meta.get("chunk_uid"), "score": score,
                "date": _date_key(meta), "source": meta.get("source"),
                "reason": "relevant_uninspected_structured_candidate",
                "available_actions": [action.type], "action": action,
            })
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
    state_markers = ("approved", "rejected", "pending", "cancelled", "closed", "validated", "decision", "accepted", "refused", "en attente", "accepte", "refuse", "valide", "annule", "cloture", "decision")
    return any(not _is_header_only(meta) and any(marker in str(meta.get("text") or "").lower() for marker in state_markers) for _, meta in items)


def _inspect(
    evidence: EvidenceSet, pool: CandidatePool, *, corpus: list[dict[str, Any]], query: str, semantics: QuerySemantics,
) -> EvidenceSufficiency:
    items = _priority_items(evidence, semantics)
    if not items:
        return EvidenceSufficiency(False, "no_retrieved_evidence")
    leads = _unresolved_leads(pool, evidence, corpus=corpus, query=query)
    if leads and semantics in {"decision", "current_state", "chronology"}:
        return EvidenceSufficiency(False, "semantic_question_has_unresolved_answer_bearing_candidate", leads[0]["action"])
    if _has_answer_bearing_content(items, semantics):
        return EvidenceSufficiency(True, "semantic_answer_bearing_evidence_available")
    for _score, meta in items:
        document_id = str(meta.get("document_id") or "")
        if not document_id or not _is_header_only(meta):
            continue
        for attachment_id in _attachment_ids(meta):
            if attachment_id not in evidence.followed_attachments and _rows_for_document(corpus, attachment_id):
                return EvidenceSufficiency(False, "relevant_email_header_has_available_attachment", RetrievalAction("FOLLOW", document_id, "attachment"))
        if document_id not in evidence.expanded_documents:
            unseen = [row for row in _rows_for_document(corpus, document_id) if str(row.get("chunk_uid")) not in evidence.seen_chunk_uids]
            if unseen:
                return EvidenceSufficiency(False, "relevant_email_header_has_unseen_document_content", RetrievalAction("EXPAND", document_id, "adjacent_chunks"))
        thread_id = _thread_id(meta)
        if thread_id and thread_id not in evidence.followed_threads:
            thread_rows = [
                row for row in corpus
                if str((row.get("document_metadata") or {}).get("thread_id") or "") == thread_id
                and str(row.get("chunk_uid")) not in evidence.seen_chunk_uids
            ]
            if thread_rows:
                return EvidenceSufficiency(False, "relevant_email_header_has_unseen_thread_evidence", RetrievalAction("FOLLOW", document_id, "same_thread"))
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
    leads = _unresolved_leads(pool, evidence, corpus=corpus, query=query)
    rounds[-1]["unresolved_leads"] = [{key: value for key, value in lead.items() if key != "action"} for lead in leads]
    rounds[-1]["sufficiency"] = {
        "sufficient": decision.sufficient, "reason": decision.reason,
        "next_action": decision.next_action.__dict__ if decision.next_action else None,
    }

    while not decision.sufficient and len(rounds) < max_rounds and decision.next_action:
        action = decision.next_action
        added = evidence.add(_execute_action(action, evidence, pool, corpus, max_expanded_chunks=max_expanded_chunks))
        round_data = {
            "round": len(rounds) + 1, "action": action.__dict__, "selected_lead": action.target_document_id,
            "lead_selection_reason": "highest-ranked unresolved structured candidate",
            "candidate_pool_count": len(pool.items), "evidence_count": len(evidence.items),
            "added_chunk_uids": [meta.get("chunk_uid") for _, meta in added],
        }
        if not added:
            decision = EvidenceSufficiency(False, "requested_action_added_no_new_evidence")
        else:
            decision = _inspect(evidence, pool, corpus=corpus, query=query, semantics=semantics)
        leads = _unresolved_leads(pool, evidence, corpus=corpus, query=query)
        round_data["unresolved_leads"] = [{key: value for key, value in lead.items() if key != "action"} for lead in leads]
        round_data["sufficiency"] = {
            "sufficient": decision.sufficient, "reason": decision.reason,
            "next_action": decision.next_action.__dict__ if decision.next_action else None,
        }
        rounds.append(round_data)

    if not decision.sufficient and len(rounds) >= max_rounds:
        decision = EvidenceSufficiency(False, "max_rounds_reached")
        rounds[-1]["sufficiency"] = {"sufficient": False, "reason": decision.reason, "next_action": None}
    return _priority_items(evidence, semantics), rounds, decision
