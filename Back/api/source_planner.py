"""Small deterministic policy layer above existing source implementations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
import unicodedata
from typing import Any, Callable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field


SourceKind = Literal["local", "email", "web", "general", "history"]
AmbiguityLevel = Literal["none", "minor", "blocking"]
SourceAnswerability = Literal["answerable", "partial", "unanswerable", "not_checked"]
RetrievalScope = Literal["none", "current_grounding", "structural_relations", "active_documents", "active_source", "global"]
NextSourceAction = Literal[
    "STOP_AND_ANSWER", "ASK_CLARIFICATION", "SEARCH_LOCAL", "SEARCH_EMAIL",
    "SEARCH_WEB", "USE_GENERAL", "ANSWER_PARTIAL", "ABSTAIN",
]


class SourcePlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: SourceKind
    priority: int = Field(default=1, ge=1, le=10)
    required: bool = False
    complementary: bool = False
    reason: str = Field(default="", max_length=240)


@dataclass(frozen=True)
class ActiveSourceContext:
    source: SourceKind | None = None
    source_ids: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    label: str | None = None
    origin: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundedConversationContext:
    """Small durable state made only from validated answer provenance."""
    active_subject: str | None = None
    primary_document_ids: tuple[str, ...] = ()
    secondary_document_ids: tuple[str, ...] = ()
    active_chunk_uids: tuple[str, ...] = ()
    active_source_type: SourceKind | None = None
    active_source_ids: tuple[str, ...] = ()
    active_email_message_id: str | None = None
    active_email_document_id: str | None = None
    active_attachment_document_ids: tuple[str, ...] = ()
    supported_claims: tuple[str, ...] = ()
    unresolved_information_needs: tuple[str, ...] = ()
    source_turn: int | None = None
    grounding_valid: bool = False

    @property
    def active_document_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.primary_document_ids, *self.secondary_document_ids)))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["active_document_ids"] = list(self.active_document_ids)
        return value


@dataclass(frozen=True)
class ConversationRetrievalDecision:
    operation: str
    retrieval_scope: RetrievalScope
    reuse_prior_grounding: bool = False
    reuse_active_documents: bool = False
    requires_structural_lookup: bool = False
    requires_global_search: bool = False
    reason: str = ""


@dataclass(frozen=True)
class SourceEvidenceResult:
    source_type: SourceKind
    source_confidence: float
    evidence_blocks: tuple[Any, ...] = ()
    supported_claims: tuple[str, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    freshness: str | None = None
    source_priority: int = 1
    answerability: SourceAnswerability = "not_checked"


@dataclass(frozen=True)
class ClaimEvidence:
    claim: str
    source_type: SourceKind
    source_id: str | None
    support_level: Literal["direct", "partial", "general"]
    citation_required: bool
    confidence: float


@dataclass
class SourcePlanExecution:
    results: list[SourceEvidenceResult] = field(default_factory=list)
    sources_checked: list[SourceKind] = field(default_factory=list)
    sources_skipped: list[dict[str, str]] = field(default_factory=list)
    next_source_action: NextSourceAction = "ABSTAIN"


def evaluate_grounding_candidate(
    *,
    mode: str | None,
    status: str | None,
    validations: dict[str, Any],
    sources: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate durable grounding from the final answer, never its execution path.

    ``orchestrator_failed`` and fallback/retry markers are deliberately absent
    from this decision: they are execution-health diagnostics, not evidence
    quality. ``sources`` must be the final reader-facing source list, not
    retrieval candidates.
    """
    terminal_mode = str(mode or "").upper()
    if terminal_mode == "CLARIFICATION":
        return {"valid": False, "rejected_reason": "clarification", "basis": "clarification"}
    if terminal_mode == "ABSTAIN" or str(status or "").casefold() == "abstained":
        return {"valid": False, "rejected_reason": "unanswerable", "basis": "unanswerable"}
    if terminal_mode == "ERROR":
        return {"valid": False, "rejected_reason": "terminal_error", "basis": "terminal_error"}

    answerability = validations.get("answerability") or {}
    answerability_status = (
        str(answerability.get("status") or answerability.get("answerability") or "") if isinstance(answerability, dict)
        else str(answerability)
    ).casefold()
    if answerability_status == "unanswerable" or validations.get("unanswerable_from_current_sources"):
        return {"valid": False, "rejected_reason": "unanswerable", "basis": "unanswerable"}

    provenance = validations.get("evidence_provenance") or {}
    if not provenance.get("evidence_sufficient"):
        return {"valid": False, "rejected_reason": "evidence_not_sufficient", "basis": "no_reliable_evidence"}
    primary_documents = tuple(str(value) for value in provenance.get("primary_document_ids") or () if value)
    if not primary_documents:
        return {"valid": False, "rejected_reason": "no_primary_document_used", "basis": "no_reliable_evidence"}

    # Final citations/sources and claim provenance are answer-bearing evidence;
    # retrieval candidates are intentionally not considered here.
    documentary_sources = [item for item in sources if isinstance(item, dict) and any(
        item.get(key) for key in ("document_id", "path", "file", "origin_path", "indexed_path")
    )]
    documentary_claims = [item for item in (validations.get("claim_sources") or []) if isinstance(item, dict)
                          and item.get("source_id") and str(item.get("source_type") or "").casefold() not in {"general", "history"}]
    if not documentary_sources and not documentary_claims:
        return {"valid": False, "rejected_reason": "no_documentary_source_used", "basis": "no_reliable_evidence"}

    evidence_documents = {str(value) for value in provenance.get("evidence_document_ids") or () if value}
    evidence_chunks = tuple(str(value) for value in provenance.get("evidence_chunk_uids") or () if value)
    if not evidence_chunks and not evidence_documents.intersection(primary_documents):
        return {"valid": False, "rejected_reason": "no_verifiable_provenance", "basis": "no_reliable_evidence"}
    return {"valid": True, "rejected_reason": None, "basis": "final_evidence_quality"}


def derive_active_source_context(history: Sequence[dict[str, Any]]) -> ActiveSourceContext:
    """Recover source continuity only from explicit prior provenance."""
    for message in reversed(history):
        if str(message.get("role")) != "assistant":
            continue
        meta = message.get("meta") or {}
        sources = meta.get("sources") or []
        provenance = meta.get("evidence_provenance") or {}
        ids = tuple(
            str(item.get("file") or item.get("title") or item.get("url") or item.get("path") or "")
            for item in sources
            if isinstance(item, dict) and (item.get("file") or item.get("title") or item.get("url") or item.get("path"))
        )
        source_document_ids = [
            str(item.get("document_id")) for item in sources
            if isinstance(item, dict) and item.get("document_id")
        ]
        documents = tuple(dict.fromkeys([
            *(str(value) for value in (provenance.get("evidence_document_ids") or []) if value),
            *source_document_ids,
        ]))
        generation_mode = str(meta.get("generation_mode") or meta.get("mode") or "")
        if ids or documents or "local" in generation_mode.lower():
            checked = {str(value) for value in (meta.get("sources_checked") or [])}
            email_paths = bool(ids) and all("email" in value.casefold() or "mail" in value.casefold() for value in ids)
            source: SourceKind = "email" if checked == {"email"} or email_paths or (
                sources and all(str(item.get("source") or item.get("type") or "") == "email" for item in sources if isinstance(item, dict))
            ) else "local"
            label = ids[0].replace("\\", "/").rsplit("/", 1)[-1] if ids else None
            return ActiveSourceContext(source, ids, documents, label, "previous_evidence_provenance")
        if "web" in generation_mode.lower():
            return ActiveSourceContext("web", ids, documents, ids[0] if ids else None, "previous_response_mode")
        checked = [str(value) for value in (meta.get("sources_checked") or [])]
        answerability = meta.get("source_answerability") or {}
        for source in checked:
            if source in {"local", "email", "web"} and answerability.get(source) in {"answerable", "partial"}:
                return ActiveSourceContext(source, ids, documents, ids[0] if ids else None, "previous_source_plan_execution")
    return ActiveSourceContext()


def derive_grounded_conversation_context(history: Sequence[dict[str, Any]]) -> GroundedConversationContext:
    """Find the last *successful* grounding, skipping invalid turns.

    This is intentionally derived from persisted history: a clarification or an
    abstention therefore cannot overwrite a previously validated document.
    """
    for index in range(len(history) - 1, -1, -1):
        message = history[index]
        if message.get("role") != "assistant":
            continue
        meta = message.get("meta") or {}
        provenance = meta.get("evidence_provenance") or {}
        # A failed candidate must never replace a previous validated grounding.
        # Legacy persisted turns have no marker and retain their old contract.
        if meta.get("grounding_candidate_valid") is False:
            continue
        if str(meta.get("mode") or "").upper() in {"CLARIFICATION", "ABSTAIN", "ERROR"}:
            continue
        answerability = meta.get("answerability") or {}
        answerability_status = str(answerability.get("status") or answerability.get("answerability") or "") if isinstance(answerability, dict) else str(answerability)
        if answerability_status.casefold() == "unanswerable" or meta.get("unanswerable_from_current_sources"):
            continue
        if not (provenance.get("evidence_sufficient") and provenance.get("evidence_chunk_uids")):
            continue
        sources = [item for item in (meta.get("sources") or []) if isinstance(item, dict)]
        claims = [str(item.get("claim")) for item in (meta.get("claim_sources") or [])
                  if isinstance(item, dict) and item.get("claim") and item.get("citation_required", True)]
        document_ids = tuple(dict.fromkeys(str(value) for value in provenance.get("evidence_document_ids", []) if value))
        primary = tuple(dict.fromkeys(str(value) for value in (
            meta.get("primary_document_ids") or provenance.get("primary_document_ids") or document_ids
        ) if value))
        secondary = tuple(value for value in document_ids if value not in primary)
        attachments = tuple(dict.fromkeys(str(item.get("attachment_document_id") or item.get("document_id"))
            for item in (meta.get("attachment_states") or []) if isinstance(item, dict) and (item.get("attachment_document_id") or item.get("document_id"))))
        source = next((str(item.get("source") or item.get("type")) for item in sources if item.get("source") or item.get("type")), None)
        if source not in {"local", "email", "web", "general", "history"}:
            source = "email" if meta.get("email_target_document_id") else "local"
        source_ids = tuple(dict.fromkeys(str(item.get("path") or item.get("file") or item.get("document_id"))
            for item in sources if item.get("path") or item.get("file") or item.get("document_id")))
        unresolved = list(meta.get("unresolved_information_needs") or [])
        if provenance.get("critical_structural_evidence_incomplete"):
            unresolved.append("structural_content")
        return GroundedConversationContext(
            active_subject=str(meta.get("retrieval_query") or message.get("content") or "")[:500] or None,
            primary_document_ids=primary, secondary_document_ids=secondary,
            active_chunk_uids=tuple(str(value) for value in provenance.get("evidence_chunk_uids", []) if value),
            active_source_type=source, active_source_ids=source_ids,
            active_email_message_id=meta.get("email_target_message_id"),
            active_email_document_id=meta.get("email_target_document_id"),
            active_attachment_document_ids=attachments, supported_claims=tuple(claims),
            unresolved_information_needs=tuple(dict.fromkeys(map(str, unresolved))),
            source_turn=index, grounding_valid=True,
        )
    return GroundedConversationContext()


def decide_conversation_retrieval(
    question: str, context: GroundedConversationContext, *, intent: str | None = None,
    reuse_previous_subject: bool = False, needs_retrieval: bool = True,
    orchestrator_failed: bool = False,
) -> ConversationRetrievalDecision:
    """Conservative local policy, independent of LLM knowledge of corpus links."""
    if not context.grounding_valid:
        return ConversationRetrievalDecision("NEW_GLOBAL_SEARCH", "global", requires_global_search=True, reason="no_valid_prior_grounding")
    text = " ".join((question or "").casefold().split())
    # Generic navigation signals: these describe a document relationship, not
    # a business domain.  They intentionally tolerate a more precise noun
    # (for example, "the order joined to the email").
    generic_relation = bool(re.search(r"\b(?:join\w*|attach\w*|annex\w*|body|page|section|table|content)\b", text))
    generic_document_ref = bool(re.search(r"\b(?:mail|e-?mail|document|pdf|file|piece|annex|table|page|section)\b", text))
    generic_pronoun_ref = bool(re.search(r"\b(?:ce|cet|cette|celui\w*|celle\w*|son|sa|ses|same|above|previous)\b", text))
    demonstrative_reference = bool(re.search(r"\b(?:ce|cet|cette|celui\w*|celle\w*|same|above|previous)\b", text))
    unresolved_attachment = bool(context.active_attachment_document_ids) and any(
        "attachment" in need.casefold() or "structural" in need.casefold()
        for need in context.unresolved_information_needs
    )
    structural_relation_candidate = (
        generic_relation and (bool(context.active_document_ids) or bool(context.active_email_document_id))
    ) or (unresolved_attachment and (generic_relation or generic_document_ref or generic_pronoun_ref)) or (
        demonstrative_reference and len(context.active_attachment_document_ids) == 1
    )
    generic_referential = (generic_document_ref and (generic_pronoun_ref or text.startswith("et "))) or (
        generic_pronoun_ref and bool(context.active_document_ids)
    )
    if intent == "web_search":
        return ConversationRetrievalDecision("NEW_GLOBAL_SEARCH", "global", requires_global_search=True, reason="explicit_source_switch")
    relation = bool(re.search(r"\b(?:pi[eè]ce jointe|document joint|annexe|attachment|corps du (?:mail|courriel)|page suivante|tableau li[eé])\b", text))
    referential = relation or bool(re.search(r"\b(?:ce|cet|cette|celui|celle|le m[eê]me|au[- ]dessus|pr[eé]c[eé]dent)\s+(?:mail|e-?mail|courriel|document|pdf|fichier|tableau)\b", text))
    transform = bool(re.search(r"\b(?:r[eé]sume|reformule|simplifie|explique.*simple|transforme|fais.{0,20}(?:mail|tableau)|donc|en gros)\b", text))
    fresh = bool(re.search(r"\b(?:plus r[eé]cent|dernier|derni[eè]re|nouveau|nouvelle|[aà] jour|fresh|latest)\b", text))
    explicit_global = bool(re.search(r"\b(?:partout|tout le corpus|recherche g[eé]n[eé]rale|tous les documents)\b", text))
    # Structural and referential compatibility are evaluated *before* planner
    # discontinuity.  A precise follow-up must not be mistaken for a subject
    # switch simply because the planner did not set reuse_previous_subject.
    if structural_relation_candidate and (context.active_attachment_document_ids or context.active_email_document_id or context.active_document_ids):
        return ConversationRetrievalDecision(
            "STRUCTURAL_LOOKUP", "structural_relations", reuse_active_documents=True,
            requires_structural_lookup=True,
            reason="unresolved_structural_need_matches_follow_up" if unresolved_attachment else "direct_relation_of_active_document",
        )
    if generic_referential and context.active_document_ids:
        return ConversationRetrievalDecision(
            "SEARCH_WITHIN_ACTIVE_DOCUMENTS", "active_documents", reuse_active_documents=True,
            reason="compatible_referential_follow_up_with_active_document",
        )
    if generic_document_ref and re.search(r"\b(?:plus|latest|recent|new|updated)\b", text) and context.active_source_type in {"email", "local"}:
        return ConversationRetrievalDecision(
            "EXPAND_WITHIN_ACTIVE_SOURCE", "active_source", reuse_prior_grounding=True,
            reason="same_subject_requires_source_expansion",
        )
    # A planner saying it cannot reuse the subject is a stronger new-topic
    # signal than lexical overlap; explicit references remain locally resolvable.
    incompatible = intent in {"general_question", "web_search"} or (not reuse_previous_subject and not referential and not transform and needs_retrieval)
    if explicit_global or incompatible:
        return ConversationRetrievalDecision("NEW_GLOBAL_SEARCH", "global", requires_global_search=True, reason="explicit_or_incompatible_subject")
    if transform or (not needs_retrieval and context.supported_claims):
        return ConversationRetrievalDecision("REUSE_GROUNDED_CONTEXT", "none", reuse_prior_grounding=True, reason="prior_supported_grounding_covers_follow_up")
    if relation and (context.active_attachment_document_ids or context.active_email_document_id or context.active_document_ids):
        return ConversationRetrievalDecision("STRUCTURAL_LOOKUP", "structural_relations", reuse_active_documents=True, requires_structural_lookup=True, reason="direct_relation_of_active_document")
    if fresh and context.active_source_type in {"email", "local"}:
        return ConversationRetrievalDecision("EXPAND_WITHIN_ACTIVE_SOURCE", "active_source", reuse_prior_grounding=True, reason="same_subject_requires_source_expansion")
    if (referential or reuse_previous_subject or orchestrator_failed) and context.active_document_ids:
        return ConversationRetrievalDecision("SEARCH_WITHIN_ACTIVE_DOCUMENTS", "active_documents", reuse_active_documents=True, reason="compatible_follow_up_with_active_documents")
    return ConversationRetrievalDecision("NEW_GLOBAL_SEARCH", "global", requires_global_search=True, reason="local_context_not_applicable")


def annotate_active_source_candidates(
    rows: Sequence[tuple[float, dict[str, Any]]], active: ActiveSourceContext,
) -> list[tuple[float, dict[str, Any]]]:
    """Mark candidates from the previously cited source without filtering.

    Retrieval scores and RRF order remain untouched; bounded evidence selection
    may then prefer a matching active document over unrelated generic results.
    """
    def canonical(value: Any) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())

    active_ids = {canonical(value) for value in (*active.source_ids, *active.document_ids) if value}
    result = []
    for score, meta in rows:
        enriched = dict(meta)
        candidate_ids = {
            canonical(enriched.get(key))
            for key in ("file", "path", "document_id")
            if enriched.get(key)
        }
        # A stored full path and a current filename still identify the same
        # document, so allow containment after punctuation normalization.
        matched = any(
            candidate == active_id or candidate in active_id or active_id in candidate
            for candidate in candidate_ids for active_id in active_ids
            if candidate and active_id
        )
        enriched["active_source_context_match"] = matched
        result.append((score, enriched))
    return result


def explicit_source_constraint(user_message: str) -> set[SourceKind] | None:
    text = user_message.casefold()
    if any(value in text for value in ("sans chercher", "sans recherche", "without searching")):
        return {"general", "history"}
    if any(value in text for value in ("sur internet", "sur le web", "search the web")):
        return {"web"}
    if any(value in text for value in ("uniquement dans mes mails", "seulement dans mes mails", "emails only")):
        return {"email"}
    if any(value in text for value in ("dans la price list", "dans le tarif", "dans mes documents")):
        return {"local"}
    return None


def normalize_source_plan(
    proposed: Sequence[SourcePlanItem], *, mode: str, intent: str,
    web_request_explicit: bool, active: ActiveSourceContext,
    explicit_sources: set[SourceKind] | None = None,
) -> list[SourcePlanItem]:
    """Apply hard UI constraints and continuity to an advisory LLM plan."""
    if mode in {"local", "web_index"}:
        allowed = {"local", "email"}
    elif mode == "web_live":
        allowed = {"web"}
    elif mode == "general":
        allowed = {"general", "history"}
    else:
        allowed = {"local", "email", "web", "general", "history"}

    if explicit_sources:
        constrained = allowed & explicit_sources
        if constrained:
            allowed = constrained
    items = [item for item in proposed if item.source in allowed]
    if not items:
        default = "web" if web_request_explicit or intent == "web_search" else "general" if intent in {"conversation", "general_question"} else active.source or "local"
        if default not in allowed:
            default = next(iter(allowed))
        items = [SourcePlanItem(source=default, priority=1, required=True, reason="deterministic mode/intent fallback")]

    # A local follow-up cannot silently become Web unless the user requested it.
    if active.source in {"local", "email"} and not web_request_explicit and intent == "refine_previous_search":
        continuity = active.source
        items = [item for item in items if item.source != "web"]
        if not any(item.source == continuity for item in items):
            items.append(SourcePlanItem(source=continuity, priority=1, required=True, reason="active conversational source continuity"))

    dedup: dict[SourceKind, SourcePlanItem] = {}
    for item in sorted(items, key=lambda value: value.priority):
        dedup.setdefault(item.source, item)
    return sorted(dedup.values(), key=lambda value: (value.priority, not value.required))


def decide_next_source_action(
    *, answerability: SourceAnswerability, clarification_needed: bool,
    current_sources_checked: Sequence[SourceKind], source_plan: Sequence[SourcePlanItem],
    max_source_expansions: int = 2,
) -> NextSourceAction:
    if clarification_needed:
        return "ASK_CLARIFICATION"
    checked = set(current_sources_checked)
    if answerability == "answerable" and not any(
        (item.required or item.complementary) and item.source not in checked
        for item in source_plan
    ):
        return "STOP_AND_ANSWER"
    if len(checked) >= max_source_expansions + 1:
        return "ANSWER_PARTIAL" if answerability == "partial" else "ABSTAIN"
    for item in source_plan:
        if item.source in checked:
            continue
        return {"local": "SEARCH_LOCAL", "email": "SEARCH_EMAIL", "web": "SEARCH_WEB", "general": "USE_GENERAL", "history": "USE_GENERAL"}[item.source]
    return "ANSWER_PARTIAL" if answerability == "partial" else "ABSTAIN"


def execute_source_plan(
    source_plan: Sequence[SourcePlanItem], executors: dict[SourceKind, Callable[[SourcePlanItem], SourceEvidenceResult]],
    *, clarification_needed: bool = False, max_source_expansions: int = 2,
) -> SourcePlanExecution:
    """Bounded executor used by the pipeline and independently unit-testable."""
    execution = SourcePlanExecution()
    if clarification_needed:
        execution.next_source_action = "ASK_CLARIFICATION"
        return execution
    for item in source_plan:
        if len(execution.sources_checked) >= max_source_expansions + 1:
            execution.sources_skipped.append({"source": item.source, "reason": "max_source_expansions_reached"})
            continue
        if execution.results and execution.results[-1].answerability == "answerable" and not (item.required or item.complementary):
            execution.sources_skipped.append({"source": item.source, "reason": "previous_source_answerable"})
            continue
        executor = executors.get(item.source)
        if not executor:
            execution.sources_skipped.append({"source": item.source, "reason": "executor_unavailable"})
            continue
        result = executor(item)
        execution.results.append(result)
        execution.sources_checked.append(item.source)
    state: SourceAnswerability = "unanswerable"
    if any(result.answerability == "answerable" for result in execution.results):
        state = "answerable"
    elif any(result.answerability == "partial" for result in execution.results):
        state = "partial"
    execution.next_source_action = decide_next_source_action(
        answerability=state, clarification_needed=False,
        current_sources_checked=execution.sources_checked, source_plan=source_plan,
        max_source_expansions=max_source_expansions,
    )
    return execution


def match_structured_values(query: str, evidence_blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Find explicit spreadsheet row/column matches already present in evidence."""
    normalized = unicodedata.normalize("NFKD", query.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    query_tokens = set(re.findall(r"[a-z]+\d+|\d+(?:\.\d+)?|[a-z]{2,}", normalized))
    query_years = {token for token in query_tokens if re.fullmatch(r"(?:19|20)\d{2}", token)}
    matches = []
    for block in evidence_blocks:
        for native in block.get("blocks") or []:
            metadata = native.get("source_metadata") or {}
            if native.get("block_type") != "table_row" and not metadata.get("structured_cells"):
                continue
            row_key = str(metadata.get("row_key") or "")
            row_digits = set(re.findall(r"\d+(?:\.\d+)?", row_key))
            if row_digits and not any(value in normalized for value in row_digits):
                continue
            for cell in metadata.get("structured_cells") or []:
                column = str(cell.get("column_key") or cell.get("column_name") or "")
                column_tokens = set(re.findall(r"[a-z]+\d+|\d+(?:\.\d+)?|[a-z]{2,}", unicodedata.normalize("NFKD", column.casefold())))
                named = {token for token in column_tokens if not token.isdigit()}
                if named and not (named & query_tokens):
                    continue
                column_years = {token for token in column_tokens if re.fullmatch(r"(?:19|20)\d{2}", token)}
                if query_years and column_years and not (query_years & column_years):
                    continue
                matches.append({
                    "sheet_name": metadata.get("sheet_name"), "table_id": metadata.get("table_id"),
                    "matched_row": row_key, "matched_column": column,
                    "matched_value": cell.get("cell_value"),
                    "value_origin": cell.get("value_origin", "explicit"),
                })
    return {
        "structured_match": bool(matches),
        "structured_match_details": matches[:10],
        "value_origin": matches[0]["value_origin"] if len(matches) == 1 else None,
        "ambiguous_value_types": len({(item.get("sheet_name"), item.get("matched_column")) for item in matches}) > 1,
    }


def structured_clarification(
    query: str, structured_match: dict[str, Any], *, enabled: bool = True,
) -> dict[str, Any] | None:
    """Return a blocking clarification when several structured values fit.

    This is deliberately schema-driven: it exposes the labels carried by the
    workbook instead of maintaining a product or price vocabulary.
    """
    if not enabled or not structured_match.get("ambiguous_value_types"):
        return None
    details = structured_match.get("structured_match_details") or []
    labels = []
    for item in details:
        label = " ".join(
            part for part in (str(item.get("sheet_name") or "").strip(), str(item.get("matched_column") or "").strip())
            if part
        )
        if label and label.casefold() not in {value.casefold() for value in labels}:
            labels.append(label)
    if len(labels) < 2:
        return None
    normalized_query = unicodedata.normalize("NFKD", query.casefold())
    normalized_query = "".join(char for char in normalized_query if not unicodedata.combining(char))
    # If a distinguishing workbook label is already stated, ambiguity has
    # been resolved by the user. Numeric-only labels (typically years) are
    # considered only when their exact value occurs in the query.
    for label in labels:
        normalized_label = unicodedata.normalize("NFKD", label.casefold())
        normalized_label = "".join(char for char in normalized_label if not unicodedata.combining(char))
        distinguishing = [
            token for token in re.findall(r"[a-z]{3,}|\d{4}", normalized_label)
            if token not in {"prix", "price", "kg2", "dn"}
        ]
        if distinguishing and all(token in normalized_query for token in distinguishing):
            return None
    choices = labels[:4]
    return {
        "clarification_needed": True,
        "clarification_reason": "multiple_structured_values_match_the_request",
        "missing_information": ["structured_value_type_or_column"],
        "ambiguity_level": "blocking",
        "clarification_question": "J'ai trouvé plusieurs valeurs possibles : " + "; ".join(choices) + ". Laquelle veux-tu ?",
    }
