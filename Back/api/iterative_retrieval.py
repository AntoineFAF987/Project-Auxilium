"""Bounded, deterministic evidence enrichment around the existing RAG search.

This module never queries FAISS/BM25, generates text, or invents identifiers.
It only follows relations already present in retrieved metadata and the loaded
corpus.  It is intentionally a finite state machine, not an agent loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
import time
from typing import Any, Callable, Literal


logger = logging.getLogger(__name__)


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
    # The first selected, high-confidence material is a stable core.  Later
    # traversal may enrich it, but must not replace it merely by being newer
    # in the traversal or by exposing another relation.
    core_chunk_uids: set[str] = field(default_factory=set)

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

    def item_representing(self, chunk_uid: str) -> tuple[float, dict[str, Any]] | None:
        """Return selected evidence that contains this native chunk.

        Retrieval fusion can represent adjacent native chunks as one evidence
        item.  A member UID is therefore allowed to be "seen" while still
        needing semantic inspection during a later structural expansion.
        """
        for score, meta in self.items:
            uid = str(meta.get("chunk_uid") or f"{meta.get('document_id')}:{meta.get('chunk_id')}")
            members = {str(value) for value in (meta.get("fused_chunk_uids") or []) if value} or {uid}
            if str(chunk_uid) in members:
                return score, meta
        return None


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
    # A header is a retrieval lead, not enough content to decide whether it
    # resolves (or changes) a state.  This avoids freezing a premature false.
    state_change_potential: Literal["true", "false", "unknown"]
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


@dataclass(frozen=True)
class DocumentProfile:
    """Document-level signals derived from the already ranked candidate pool."""
    document_id: str
    best_score: float
    mean_score: float
    top_chunk_count: int
    query_variant_count: int
    relative_score: float
    anchor_score: float
    is_anchor: bool


def _document_profiles(pool: CandidatePool) -> dict[str, DocumentProfile]:
    """Aggregate chunk ranking signals without making filenames semantic rules."""
    if not pool.items:
        return {}
    best = max(float(score) for score, _ in pool.items) or 1.0
    grouped: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    for score, meta in pool.items:
        document_id = str(meta.get("document_id") or "")
        if document_id:
            grouped.setdefault(document_id, []).append((float(score), meta))
    profiles: dict[str, DocumentProfile] = {}
    for document_id, rows in grouped.items():
        scores = [score for score, _ in rows]
        variants = {kind for _, meta in rows for kind in (meta.get("retrieved_by") or [])}
        # Several high-ranked chunks and agreement between query variants are
        # independent support for a document.  This is intentionally relative
        # to the current result set rather than a corpus-wide fixed threshold.
        top_chunks = sum(score / best >= 0.80 for score in scores)
        relative = max(scores) / best
        variant_count = len(variants)
        anchor_score = relative + min(top_chunks - 1, 2) * 0.10 + min(variant_count - 1, 2) * 0.05
        other_scores = [max(values) for key, values in ((key, [score for score, _ in value]) for key, value in grouped.items()) if key != document_id]
        next_best = max(other_scores, default=0.0)
        # An anchor is deliberately scarce: a corroborated top result or a
        # single result clearly separated from every other document.
        is_anchor = relative == 1.0 and (top_chunks >= 2 or variant_count >= 2 or next_best / best < 0.75)
        profiles[document_id] = DocumentProfile(
            document_id, max(scores), sum(scores) / len(scores), top_chunks,
            variant_count, relative, anchor_score, is_anchor,
        )
    return profiles


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


def _row_by_uid(corpus: list[dict[str, Any]], chunk_uid: str | None) -> dict[str, Any] | None:
    """Resolve an indexed neighbour without relying on search or row order."""
    if not chunk_uid:
        return None
    return next((row for row in corpus if str(row.get("chunk_uid")) == str(chunk_uid)), None)


def _block_types(meta: dict[str, Any]) -> set[str]:
    return {
        str(block.get("block_type")) for block in (meta.get("blocks") or [])
        if isinstance(block, dict)
    }


def _is_email_body(meta: dict[str, Any]) -> bool:
    return meta.get("source") == "email" and "email_body" in _block_types(meta)


def _requires_email_content(query: str, semantics: QuerySemantics) -> bool:
    """Whether an email header must be treated as a lead rather than evidence.

    This intentionally models the request category, not any particular subject
    line or email. Decision/state questions are content questions too.
    """
    if semantics in {"decision", "current_state"}:
        return True
    text = query.casefold()
    mentions_email = bool(re.search(r"\b(?:mail|email|e-mail|courriel)\b", text))
    asks_content = any(phrase in text for phrase in (
        "que dit", "qu'est-ce qu", "qu’ est-ce qu", "contenu", "contient",
        "regarde", "etudie", "étudie", "lire", "lis ", "lecture",
    ))
    return mentions_email and asks_content


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
        "final", "finally", "decision", "décision", "status", "statut",
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


def _title_text(meta: dict[str, Any]) -> str:
    metadata = meta.get("document_metadata") or {}
    return " ".join((str(meta.get("title") or ""), str(metadata.get("subject") or ""), str(meta.get("text") or "")[:400]))


def _topic_relation(
    meta: dict[str, Any], query: str, semantics: QuerySemantics, *,
    chunk_text_only: bool = False,
) -> tuple[Literal["direct_subject_match", "state_resolution_context", "weak_selected_evidence"], Literal["true", "false", "unknown"]]:
    """Return semantic fit and state-change potential for a retrieval item.

    Document metadata is useful while selecting a document lead: an email
    subject can identify a related message before its body is loaded. It is
    not evidence about every chunk of that message. ``chunk_text_only`` is
    required for post-EXPAND re-evaluation, where this precise chunk may be
    promoted to the generation context.

    Shared names, organisations and location labels do not establish topical
    relation by themselves.  They need an accompanying non-entity topic term.
    """
    candidate_text = str(meta.get("text") or "") if chunk_text_only else _title_text(meta)
    query_terms, candidate_terms = _terms(query), _terms(candidate_text)
    shared = query_terms & candidate_terms
    proper_shared = _proper_terms(query) & _proper_terms(candidate_text)
    non_entity_shared = shared - proper_shared
    candidate_topics = candidate_terms - proper_shared
    query_topics = query_terms - proper_shared
    competing_topic = bool(candidate_topics - query_topics) and not non_entity_shared
    # A multiword entity/label ("Project Orion", a product code plus name)
    # is a subject in its own right; one name alone remains insufficient.
    direct = non_entity_shared or (semantics in {"fact_lookup", "general_document_question"} and len(proper_shared) >= 2)
    if direct:
        # A header can nominate a document, but cannot settle a factual state.
        return "direct_subject_match", "unknown" if _is_header_only(meta) else "true"
    if semantics in {"current_state", "decision"} and not competing_topic and proper_shared:
        return "state_resolution_context", "unknown" if _is_header_only(meta) else "true"
    return "weak_selected_evidence", "false"


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
    Literal["true", "false", "unknown"],
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
    elif newer and semantics in {"current_state", "decision"} and state_change_potential != "false":
        temporal_relation = "newer_than_existing_state_evidence"
    elif newer:
        temporal_relation = "recent_relative_to_selected_evidence"
    else:
        temporal_relation = "not_newer"

    if semantics in {"current_state", "decision"} and semantic_fit != "weak_selected_evidence" and newer and state_change_potential != "false":
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
    profiles = _document_profiles(pool)
    for score, meta in pool.items:
        uid = str(meta.get("chunk_uid") or "")
        if uid in evidence.seen_chunk_uids or not _candidate_is_related(meta, query, semantics):
            continue
        profile = profiles.get(str(meta.get("document_id") or ""))
        # Weak, single-hit candidates are incidental leads, not a reason to
        # spend a round. A potentially superseding state lead remains eligible
        # only for temporal/decision questions and only when it is topical.
        fit, state_change = _topic_relation(meta, query, semantics)
        critical_temporal = semantics in {"current_state", "decision"} and fit != "weak_selected_evidence" and state_change != "false"
        if profile and profile.relative_score < 0.35 and profile.query_variant_count < 2 and not critical_temporal:
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
    core = [item for item in evidence.items if str(item[1].get("chunk_uid")) in evidence.core_chunk_uids]
    optional = [item for item in evidence.items if str(item[1].get("chunk_uid")) not in evidence.core_chunk_uids]
    if semantics not in {"current_state", "decision"}:
        return core + optional
    # State/decision queries favour the latest answer-bearing body. A header
    # remains useful metadata, but cannot displace an expanded body from the
    # same document in the final-context budget.
    body_documents = {
        str(meta.get("document_id") or "") for _, meta in evidence.items if not _is_header_only(meta)
    }
    def rank(item: tuple[float, dict[str, Any]]) -> tuple[int, str, float]:
        score, meta = item
        redundant_header = _is_header_only(meta) and str(meta.get("document_id") or "") in body_documents
        return (
            0 if redundant_header else 1,
            _date_key(meta),
            score,
        )
    return sorted(core, key=rank, reverse=True) + sorted(optional, key=rank, reverse=True)


def _has_answer_bearing_content(items: list[tuple[float, dict[str, Any]]], semantics: QuerySemantics, query: str) -> bool:
    # Content structurally expanded from a directly selected document inherits
    # that document's topical support. This avoids requiring every adjacent
    # body chunk to repeat the query wording verbatim.
    directly_supported_documents = {
        str(meta.get("document_id") or "") for _, meta in items
        if _candidate_is_related(meta, query, semantics)
    }
    structurally_supported_body = any(
        not _is_header_only(meta) and str(meta.get("document_id") or "") in directly_supported_documents
        and "attached" not in str(meta.get("text") or "").casefold()
        for _, meta in items
    )
    requested_email_content = _requires_email_content(query, semantics) and any(
        _is_email_body(meta) for _, meta in items
    )
    if semantics in {"fact_lookup", "general_document_question"}:
        return structurally_supported_body or requested_email_content
    if semantics == "chronology":
        return len({_date_key(meta)[:10] for _, meta in items if _date_key(meta)}) >= 2
    non_header_bodies = [meta for _, meta in items if not _is_header_only(meta)]
    # Retrieval establishes that a directly-related body is usable evidence;
    # interpreting its outcome belongs to semantic generation, never a
    # hand-maintained vocabulary of business-state words.
    return requested_email_content or structurally_supported_body or bool(non_header_bodies)


def _inspect(
    evidence: EvidenceSet, pool: CandidatePool, *, corpus: list[dict[str, Any]], query: str, semantics: QuerySemantics,
) -> EvidenceSufficiency:
    items = _priority_items(evidence, semantics)
    if not items:
        return EvidenceSufficiency(False, "no_retrieved_evidence")
    gaps = _evidence_gaps(evidence, corpus=corpus, query=query, semantics=semantics)
    leads = _candidate_leads(pool, evidence, corpus=corpus, query=query, semantics=semantics)
    # First complete selected *critical* partial evidence. A header is a lead, not the
    # document's answer-bearing content.
    expansion_gaps = [gap for gap in gaps if gap.available_actions and gap.available_actions[0].type == "EXPAND"]
    answer_available = _has_answer_bearing_content(items, semantics, query)
    selected_header_requires_body = any(
        gap.gap_type == "email_header_only" for gap in expansion_gaps
    ) and _requires_email_content(query, semantics)
    critical_gaps = [gap for gap in expansion_gaps if (
        selected_header_requires_body
        or gap.priority_class in {"potential_newer_state_resolution", "recent_directly_related", "temporal_coverage"}
        # An explicitly state-bearing document lead is critical for a decision
        # request even when its metadata has no usable chronological ordering.
        or (semantics == "decision" and gap.semantic_fit == "direct_subject_match" and gap.state_change_potential != "false")
    )]
    evidence_dates = [_date_key(meta) for _, meta in items if _date_key(meta)]
    newer_state_leads = [
        lead for lead in leads
        if semantics in {"current_state", "decision"}
        and lead.date and (not evidence_dates or lead.date > max(evidence_dates))
    ]
    unresolved_attachments = [
        gap for gap in gaps
        if any(action.type == "FOLLOW" and action.relation == "attachment" for action in gap.available_actions)
        and not any(action.type == "EXPAND" for action in gap.available_actions)
    ]
    if semantics == "current_state":
        dated = [(_date_key(meta), meta) for _, meta in items if _date_key(meta)]
        if dated:
            latest_date = max(date for date, _ in dated)
            latest_has_content = any(date == latest_date and not _is_header_only(meta) for date, meta in dated)
            latest_documents = {str(meta.get("document_id") or "") for date, meta in dated if date == latest_date}
            if latest_has_content or bool(latest_documents & evidence.expanded_documents):
                answer_available = True
    if newer_state_leads:
        # A topically related later document can supersede an older answer.
        # Its content must be read before declaring the existing history final.
        return EvidenceSufficiency(False, "newer_related_candidate_requires_resolution", newer_state_leads[0].available_actions[0])
    if semantics == "decision" and unresolved_attachments:
        # A declared attachment is structured answer-bearing material.  Do not
        # silently treat an email's surrounding prose as a substitute for it.
        attachment_action = next(action for action in unresolved_attachments[0].available_actions if action.relation == "attachment")
        return EvidenceSufficiency(False, "declared_attachment_requires_resolution", attachment_action)
    if answer_available and not selected_header_requires_body:
        # A resolved answer is sufficient even if optional traversal remains.
        # Criticality is assessed before answer availability only when the
        # selected evidence is still a header that requires its body.
        return EvidenceSufficiency(True, "answerable_core_evidence_optional_expansions_remaining" if (gaps or leads) else "semantic_answer_bearing_evidence_available")
    if critical_gaps and (semantics in {"decision", "current_state", "chronology"} or selected_header_requires_body) and (
        not answer_available
        or selected_header_requires_body
        or bool(critical_gaps)
    ):
        return EvidenceSufficiency(False, "critical_gap_requires_resolution", critical_gaps[0].available_actions[0])
    if answer_available:
        # Optional relations and speculative candidate leads are deliberately
        # not evidence gaps: answerability and explore-ability are separate.
        return EvidenceSufficiency(True, "answerable_core_evidence_optional_expansions_remaining" if (gaps or leads) else "semantic_answer_bearing_evidence_available")
    if leads and semantics in {"fact_lookup", "general_document_question", "decision", "current_state", "chronology"}:
        return EvidenceSufficiency(False, "critical_answer_bearing_candidate_missing", leads[0].available_actions[0])
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
        origin = pool.by_document_id(action.target_document_id) or next(
            (meta for _, meta in evidence.items if str(meta.get("document_id")) == action.target_document_id), None,
        )
        # Email expansion is deliberately structural: start at the selected
        # header's next_chunk_uid and follow adjacent email_body chunks.  This
        # cannot drift to a generic vector-search result from another email.
        if origin and _is_header_only(origin):
            logger.info("rag_email_header_detected document_id=%s chunk_uid=%s next_chunk_uid=%s",
                        action.target_document_id, origin.get("chunk_uid"), origin.get("next_chunk_uid"))
            rows: list[dict[str, Any]] = []
            current = _row_by_uid(corpus, origin.get("next_chunk_uid"))
            while current and len(rows) < max_expanded_chunks:
                if str(current.get("document_id")) != action.target_document_id or not _is_email_body(current):
                    break
                rows.append(current)
                current = _row_by_uid(corpus, current.get("next_chunk_uid"))
            # Backward-compatible structural fallback for an already-loaded
            # legacy corpus lacking neighbour UIDs: only immediate later
            # email_body rows from this very document are eligible.
            if not rows and not origin.get("next_chunk_uid"):
                origin_order = int(origin.get("order") or origin.get("chunk_id") or 0)
                candidates = sorted(
                    _rows_for_document(corpus, action.target_document_id),
                    key=lambda row: int(row.get("order") or row.get("chunk_id") or 0),
                )
                rows = [
                    row for row in candidates
                    if int(row.get("order") or row.get("chunk_id") or 0) > origin_order
                    and str(row.get("chunk_uid")) not in evidence.seen_chunk_uids
                    and _is_email_body(row)
                ][:max_expanded_chunks]
            if rows:
                logger.info("rag_email_expand_added document_id=%s chunk_uids=%s body_found=true",
                            action.target_document_id, [row.get("chunk_uid") for row in rows])
                # Keep structurally required content inside the final context
                # budget beside its retrieved header, rather than demoting it
                # behind unrelated search hits.
                origin_score = next((
                    score for score, meta in evidence.items
                    if str(meta.get("chunk_uid")) == str(origin.get("chunk_uid"))
                ), 0.0)
                return [(float(origin_score) - (index + 1) * 1e-6, row) for index, row in enumerate(rows)]
            logger.info("rag_email_expand_failed document_id=%s reason=%s",
                        action.target_document_id,
                        "next_chunk_uid_missing_or_not_email_body" if origin.get("next_chunk_uid") else "header_has_no_next_chunk_uid")
            return []
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
    on_action: Callable[[RetrievalAction], None] | None = None,
) -> tuple[list[tuple[float, dict[str, Any]]], list[dict[str, Any]], EvidenceSufficiency]:
    """Enrich evidence using only valid, unseen, structured relations."""
    pool = CandidatePool(tuple(candidate_pool))
    evidence = EvidenceSet()
    initial_added = evidence.add(initial_evidence)
    profiles = _document_profiles(pool)
    # Freeze only material supported by a dominant document or close to the
    # best ranked result.  Individual chunks remain available; this marker is
    # solely a document-level decision layer.
    for score, meta in initial_added:
        uid = str(meta.get("chunk_uid") or "")
        profile = profiles.get(str(meta.get("document_id") or ""))
        best = max((float(value) for value, _ in initial_added), default=1.0) or 1.0
        if profile and profile.is_anchor or float(score) / best >= 0.82:
            evidence.core_chunk_uids.add(uid)
    selection_started = time.perf_counter()
    rounds: list[dict[str, Any]] = [{
        "round": 1, "action": {"type": "SEARCH", "query": query},
        "candidate_pool_count": len(pool.items), "candidate_pool": pool.trace_items(),
        "evidence_count": len(evidence.items),
        "added_chunk_uids": [meta.get("chunk_uid") for _, meta in initial_added],
        "anchor_documents": [profile.__dict__ for profile in profiles.values() if profile.is_anchor],
        "core_evidence_chunk_uids": sorted(evidence.core_chunk_uids),
    }]
    decision = _inspect(evidence, pool, corpus=corpus, query=query, semantics=semantics)
    gaps = _evidence_gaps(evidence, corpus=corpus, query=query, semantics=semantics)
    leads = _candidate_leads(pool, evidence, corpus=corpus, query=query, semantics=semantics)
    rounds[-1]["evidence_gaps"] = _trace_gaps(gaps)
    rounds[-1]["state_change_candidates"] = [gap.document_id for gap in gaps if gap.state_change_potential != "false" and gap.semantic_fit != "weak_selected_evidence"]
    rounds[-1]["candidate_leads"] = _trace_leads(leads)
    # Kept for trace compatibility; it now means unselected CandidateLeads.
    rounds[-1]["unresolved_leads"] = rounds[-1]["candidate_leads"]
    rounds[-1].update(_lead_trace_fields(decision.next_action, gaps, leads))
    rounds[-1]["sufficiency"] = {
        "sufficient": decision.sufficient, "reason": decision.reason,
        "next_action": decision.next_action.__dict__ if decision.next_action else None,
    }
    rounds[-1]["evidence_selection_ms"] = round((time.perf_counter() - selection_started) * 1000, 1)

    while not decision.sufficient and len(rounds) < max_rounds and decision.next_action:
        action = decision.next_action
        selected_fields = _lead_trace_fields(action, gaps, leads)
        if on_action:
            on_action(action)
        expand_started = time.perf_counter()
        expanded_rows = _execute_action(
            action, evidence, pool, corpus, max_expanded_chunks=max_expanded_chunks,
        )
        added = evidence.add(expanded_rows)
        promoted: list[str] = []
        rejected: list[dict[str, Any]] = []
        stateful_origin = next((
            gap for gap in gaps
            if gap.document_id == action.target_document_id
            and gap.state_change_potential != "false"
            and gap.semantic_fit != "weak_selected_evidence"
        ), None)
        reevaluated_chunks: list[dict[str, Any]] = []
        if action.type == "EXPAND":
            # Re-evaluate the newly loaded content. The selected header only
            # established a topical, temporal lead; structural lineage makes
            # its non-header continuation a candidate, not passive optional
            # material. This is intentionally independent of domain wording.
            # Inspect every bounded structural neighbour, including a native
            # chunk already represented inside a fused evidence item.  "Seen"
            # means de-duplicated in EvidenceSet; it must not mean "already
            # semantically evaluated as an answer-bearing chunk".
            for _score, raw_meta in expanded_rows:
                uid = str(raw_meta.get("chunk_uid") or "")
                represented = evidence.item_representing(uid) if uid else None
                meta = represented[1] if represented else raw_meta
                # A native block may span multiple indexed chunks. Do not let
                # its text, or its email subject/title, make a signature
                # inherit a decision stated in a previous chunk.
                fit, potential = _topic_relation(
                    raw_meta, query, semantics, chunk_text_only=True,
                )
                eligible = (
                    semantics in {"current_state", "decision"}
                    and stateful_origin is not None
                    and not _is_header_only(meta)
                    and potential != "false"
                )
                if eligible:
                    promotion_reason = "expanded_related_content"
                elif stateful_origin is None:
                    promotion_reason = "expanded_origin_not_a_related_state_candidate"
                elif _is_header_only(meta):
                    promotion_reason = "expanded_chunk_is_header_only"
                else:
                    promotion_reason = "expanded_content_not_answer_bearing"
                reevaluated_chunks.append({
                    "chunk_uid": uid,
                    "evaluated_text_source": "chunk.text",
                    "already_represented_in_evidence": represented is not None and not any(
                        str(added_meta.get("chunk_uid") or "") == uid for _, added_meta in added
                    ),
                    "semantic_fit": fit,
                    "state_change_potential_before_expand": stateful_origin.state_change_potential if stateful_origin else None,
                    "state_change_potential_after_expand": potential,
                    "promotion_reason": promotion_reason,
                })
                if uid and eligible:
                    # Evidence-local trace marker. It does not change retrieval
                    # scores, corpus metadata, or final-context ordering.
                    meta["state_change_promoted"] = True
                    promoted_members = list(meta.get("state_change_promoted_member_chunk_uids") or [])
                    if uid not in promoted_members:
                        meta["state_change_promoted_member_chunk_uids"] = [*promoted_members, uid]
                    evidence.core_chunk_uids.add(uid)
                    if represented:
                        evidence.core_chunk_uids.add(str(meta.get("chunk_uid") or ""))
                    promoted.append(uid)
                else:
                    rejected.append({
                        "chunk_uid": uid,
                        "reason": promotion_reason,
                        "semantic_fit": fit,
                        "state_change_potential": potential,
                    })
        round_data = {
            "round": len(rounds) + 1, "action": action.__dict__,
            "candidate_pool_count": len(pool.items), "evidence_count": len(evidence.items),
            "added_chunk_uids": [meta.get("chunk_uid") for _, meta in added],
            "expand_ms": round((time.perf_counter() - expand_started) * 1000, 1),
            "state_change_promoted_chunks": promoted,
            "expanded_document_chunks_considered": [meta.get("chunk_uid") for _, meta in expanded_rows],
            "expanded_document_chunks_promoted": promoted,
            "expanded_document_chunks_rejected": rejected,
            "expanded_chunk_reevaluation": reevaluated_chunks,
        }
        if not added and not promoted:
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
        if semantics == "current_state" and evidence.expanded_documents:
            decision = EvidenceSufficiency(True, "latest_state_evidence_inspected")
        else:
            decision = EvidenceSufficiency(False, "max_rounds_reached")
        rounds[-1]["sufficiency"] = {"sufficient": decision.sufficient, "reason": decision.reason, "next_action": None}
    final_items = _priority_items(evidence, semantics)
    final_uids = {str(meta.get("chunk_uid") or "") for _, meta in final_items}
    final_member_uids = {
        str(member) for _, meta in final_items
        for member in (meta.get("fused_chunk_uids") or [meta.get("chunk_uid")])
        if member
    }
    rounds[-1]["optional_evidence_chunk_uids"] = sorted(final_uids - evidence.core_chunk_uids)
    rounds[-1]["core_evidence_chunk_uids"] = sorted(evidence.core_chunk_uids)
    rounds[-1]["final_evidence_member_chunk_uids"] = sorted(final_member_uids)
    rounds[-1]["latest_relevant_evidence_date"] = max((_date_key(meta) for _, meta in final_items if _date_key(meta)), default=None)
    rounds[-1]["dropped_evidence"] = [
        {"chunk_uid": str(meta.get("chunk_uid") or ""), "reason": "weak_relative_candidate_not_expanded"}
        for score, meta in pool.items
        if str(meta.get("chunk_uid") or "") not in final_uids
        and (profiles.get(str(meta.get("document_id") or "")) and profiles[str(meta.get("document_id") or "")].relative_score < 0.35)
    ]
    return final_items, rounds, decision
