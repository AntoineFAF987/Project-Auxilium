"""One-shot planning for the answer pipeline.

This module deliberately does not retrieve documents or produce user-facing text.
It turns a compact conversational view into a validated execution plan.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import date as CalendarDate, datetime, timezone
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, field_validator, model_validator


Intent = Literal["conversation", "document_question", "refine_previous_search", "general_question", "web_search"]
ResponseStrategy = Literal["answer", "ask_for_missing_information", "general_answer"]
QuerySemantics = Literal["fact_lookup", "current_state", "decision", "chronology", "comparison", "procedure", "general_document_question"]
SourceType = Literal["email", "pdf", "file"]
TemporalMode = Literal["exact", "range", "recent", "before", "after"]
ConstraintProvenance = Literal["explicit", "inferred"]


class MetadataConstraint(BaseModel):
    """A metadata value and how it was obtained.

    A bare legacy string is accepted as inferred so upgrading never turns an
    old plan into a newly-authoritative filter.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)
    value: str = Field(min_length=1, max_length=200)
    provenance: ConstraintProvenance = "inferred"

    @model_validator(mode="before")
    @classmethod
    def coerce_legacy_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"value": value, "provenance": "inferred"}
        return value


class TemporalConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: TemporalMode
    date: Optional[CalendarDate] = None
    start_date: Optional[CalendarDate] = None
    end_date: Optional[CalendarDate] = None
    strength: Literal["normal", "strong"] = "normal"
    provenance: ConstraintProvenance = "inferred"

    @model_validator(mode="after")
    def validate_dates(self) -> "TemporalConstraint":
        if self.mode in {"exact", "before", "after"} and self.date is None:
            raise ValueError("date is required for this temporal mode")
        if self.mode == "range" and (self.start_date is None or self.end_date is None):
            raise ValueError("start_date and end_date are required for range")
        return self


class MetadataConstraints(BaseModel):
    """Only fields present in current email/document index metadata."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    sender: Optional[MetadataConstraint] = None
    recipient: Optional[MetadataConstraint] = None
    thread: Optional[MetadataConstraint] = None
    document_type: Optional[MetadataConstraint] = None


class OrchestrationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    intent: Intent
    needs_retrieval: bool
    retrieval_query: Optional[str] = Field(default=None, min_length=3, max_length=400)
    use_history: bool = False
    reuse_previous_subject: bool = False
    source_types: list[SourceType] = Field(default_factory=list, max_length=3)
    # A source type inferred by the model is useful diagnostic context but is
    # never allowed to narrow the candidate set.
    source_type_provenance: dict[SourceType, ConstraintProvenance] = Field(default_factory=dict)
    temporal_constraints: list[TemporalConstraint] = Field(default_factory=list, max_length=2)
    metadata_constraints: MetadataConstraints = Field(default_factory=MetadataConstraints)
    query_semantics: QuerySemantics = "general_document_question"
    response_strategy: ResponseStrategy
    _raw_model_output: str | None = PrivateAttr(default=None)

    @field_validator("source_types", "temporal_constraints", mode="before")
    @classmethod
    def normalize_null_lists(cls, value: Any) -> Any:
        """LLMs commonly emit null for an empty optional collection."""
        return [] if value is None else value

    @field_validator("source_type_provenance", mode="before")
    @classmethod
    def normalize_null_mapping(cls, value: Any) -> Any:
        return {} if value is None else value

    @field_validator("metadata_constraints", mode="before")
    @classmethod
    def normalize_null_metadata(cls, value: Any) -> Any:
        return {} if value is None else value

    @model_validator(mode="after")
    def validate_execution_contract(self) -> "OrchestrationPlan":
        if self.needs_retrieval and not self.retrieval_query:
            raise ValueError("retrieval_query is required when retrieval is needed")
        if not self.needs_retrieval and (self.retrieval_query or self.source_types or self.temporal_constraints):
            raise ValueError("retrieval fields are forbidden when retrieval is not needed")
        if self.intent == "conversation" and self.needs_retrieval:
            raise ValueError("conversation cannot request retrieval")
        return self


SYSTEM_PROMPT = """You are Auxilium's internal orchestrator. You are not the final assistant and never answer the user. Return one JSON object only matching the schema.

Auxilium is a documentary and conversational assistant. Its core capability is searching the user's indexed local documents and indexed emails, using recent conversation history, temporal constraints, and available metadata. It can answer generally only when those sources are not relevant. Prefer the user's own information whenever it could contain the answer.

Your mission is to choose strategy: retrieval need, standalone retrieval query, history reuse, source scope, temporal/metadata hints, and response strategy. You never retrieve, inspect or cite documents, invent facts, simulate an action, decide evidence quality, or bypass safeguards. The RAG, evidence_mode, and final generator retain those responsibilities.

Also set query_semantics: use current_state or decision when the user asks for a current status, outcome, or change; otherwise choose the closest supported general documentary need.

Capabilities: indexed local documents; indexed emails; recent history; live Web search when available; supported metadata and time constraints; general conversation. Limits: no live mailbox outside synchronized/indexed data, no external case files, no unconnected source, no external action. No result in a retrieval is not lack of access to indexed sources.

Choose only: conversation, document_question, refine_previous_search, general_question, web_search. Use document retrieval for information that could reasonably be in the user's or organisation's sources: personal or business status, decision, request, validation, correspondence, project, internal procedure, technical reference, product, or recent internal event. Use web_search when the user explicitly asks to search the Web/internet, or when fresh public information is needed. For web_search set needs_retrieval=true and make retrieval_query a concise standalone Web query. When a source-switch request such as "search the web" refers to an active subject, set use_history=true and reuse_previous_subject=true; retrieval_query must retain that subject rather than merely repeating the source-switch instruction. When uncertain between document_question and general_question, prefer document_question. Do not retrieve for obvious social conversation, politeness, incomplete introductions, creative requests, or general explanations independent of local sources.

For a follow-up or explicit request to search documents/emails: if an active subject is recoverable, use intent=refine_previous_search, needs_retrieval=true, use_history=true, reuse_previous_subject=true. Make retrieval_query a concise standalone search query, never an instruction to another model: preserve important entities, nouns, technical terms and relation wording; do not add generic search boilerplate such as finding any relevant decision, validation, rejection or correspondence. Narrow source_types when named. If no subject is recoverable, ask for missing information instead of a vague search. Dates and metadata refine relevance; never invent them.

Every source, temporal, or metadata constraint has provenance. `source_type_provenance` maps each source type to "explicit" or "inferred"; each temporal or metadata constraint has its own `provenance`. Set "explicit" only when the user actually asks to limit or select on that property. A person, organisation, date, or product merely mentioned as the subject is not a sender, recipient, source, document-type, or date filter: preserve it in retrieval_query instead and mark any optional hint "inferred". Inferred constraints are hints, never restrictions.

Examples (conceptual, JSON shape abbreviated):
- "Hello" -> conversation, needs_retrieval=false, response_strategy=general_answer.
- "I received a question" -> conversation, needs_retrieval=false, response_strategy=ask_for_missing_information.
- "Can model X be adapted for a harsh environment?" -> document_question, needs_retrieval=true.
- "Was my request approved?" -> document_question, needs_retrieval=true.
- "Search recent emails instead" with an active subject -> refine_previous_search, needs_retrieval=true, use_history=true, reuse_previous_subject=true, source_types=["email"], temporal_constraints=[{"mode":"recent"}].
- "Search the web instead" with an active subject -> web_search, needs_retrieval=true, use_history=true, reuse_previous_subject=true.
- "Explain REST APIs" -> general_question, needs_retrieval=false.
"""


def compact_history(history: list[dict[str, Any]], *, max_messages: int = 6) -> dict[str, Any]:
    """Recent turns and candidate user subjects, never the full transcript."""
    recent = [
        {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))[:500]}
        for item in history[-max_messages:]
        if isinstance(item, dict) and str(item.get("content", "")).strip()
    ]
    subjects = [item["content"][:300] for item in recent if item["role"] == "user"]
    return {
        "recent_turns": recent,
        # Candidates intentionally retain the substantive turn before a terse
        # follow-up; the planner, not a phrase list, resolves the reference.
        "active_subject_candidates": subjects[-3:],
    }


def build_prompt(question: str, history: list[dict[str, Any]]) -> str:
    view = compact_history(history)
    schema = OrchestrationPlan.model_json_schema()
    return (
        f"{SYSTEM_PROMPT}\nCURRENT_DATE_UTC: {datetime.now(timezone.utc).date().isoformat()}\n"
        f"COMPACT_HISTORY: {json.dumps(view, ensure_ascii=False)}\n"
        f"USER_MESSAGE: {question}\n"
        f"JSON_SCHEMA: {json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}\n"
        "Return JSON only."
    )


def build_json_repair_prompt(question: str, history: list[dict[str, Any]]) -> str:
    """Small retry prompt used only after the planner emitted invalid JSON."""
    view = compact_history(history)
    schema = OrchestrationPlan.model_json_schema()
    return (
        "Return ONLY one valid JSON object matching this schema. No prose, no markdown, no explanation.\n"
        f"USER_MESSAGE: {question}\n"
        f"COMPACT_HISTORY: {json.dumps(view, ensure_ascii=False)}\n"
        f"JSON_SCHEMA: {json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
    )


def _parse_plan(raw: str | None) -> OrchestrationPlan:
    """Validate planner output while retaining it for local diagnostics."""
    try:
        parsed = json.loads((raw or "").strip())
        if not isinstance(parsed, dict):
            raise ValueError("orchestration output must be a JSON object")
        plan = OrchestrationPlan.model_validate(parsed)
        plan._raw_model_output = raw or ""
        return plan
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        details = exc.errors() if isinstance(exc, ValidationError) else [{"message": str(exc)}]
        raise OrchestrationPlanOutputError(raw_model_output=raw or "", validation_error_details=details) from exc


def plan_once(question: str, history: list[dict[str, Any]], call_llm: Callable[..., str], *, model: str | None, timeout: int) -> OrchestrationPlan:
    """Execute exactly one LLM call; malformed output raises for the caller's safe fallback."""
    raw = call_llm(build_prompt(question, history), context_text="", history=[], model=model, timeout=timeout, max_tokens=420)
    return _parse_plan(raw)


def plan_json_retry(question: str, history: list[dict[str, Any]], call_llm: Callable[..., str], *, model: str | None, timeout: int) -> OrchestrationPlan:
    """One compact repair attempt; callers decide whether it is appropriate."""
    raw = call_llm(build_json_repair_prompt(question, history), context_text="", history=[], model=model, timeout=timeout, max_tokens=420)
    return _parse_plan(raw)


class OrchestrationPlanOutputError(ValueError):
    """Safe fallback error retaining local-only diagnostic details."""

    def __init__(self, *, raw_model_output: str, validation_error_details: list[dict[str, Any]]):
        super().__init__("invalid orchestration output")
        self.raw_model_output = raw_model_output
        self.validation_error_details = validation_error_details


@dataclass(frozen=True)
class SanitizedPlan:
    """Execution-safe plan plus a transparent audit of constraint handling."""
    plan: OrchestrationPlan
    hard_filters: list[dict[str, Any]]
    soft_preferences: list[dict[str, Any]]
    removed_constraints: list[dict[str, Any]]


def _normalized_text(value: str) -> str:
    return " ".join(
        "".join(char for char in unicodedata.normalize("NFKD", value.lower()) if not unicodedata.combining(char)).split()
    )


def _value_pattern(value: str) -> str:
    return r"\b" + r"\s+".join(re.escape(token) for token in _normalized_text(value).split()) + r"\b"


def _has_explicit_source_scope(text: str, source: str) -> bool:
    terms = {
        "email": r"(?:mail|mails|email|emails|courriel|courriels)",
        "pdf": r"(?:pdf|pdfs)",
        "file": r"(?:fichier|fichiers|document|documents|file|files)",
    }[source]
    # A source must be tied to a search/scope relation; merely mentioning a
    # document or email does not narrow retrieval.
    scope = r"(?:dans|parmi|uniquement|seulement|exclusivement|cherche(?:r)?|recherche(?:r)?|regarde(?:r)?|consulte(?:r)?|verifie(?:r)?|search|look|only|within|from)"
    return bool(re.search(rf"\b{scope}\b(?:\s+\w+){{0,5}}\s+{terms}\b|{terms}\b(?:\s+\w+){{0,5}}\s+\b{scope}\b", text))


def _date_is_mentioned(text: str, constraint: TemporalConstraint) -> bool:
    target = constraint.date or constraint.start_date or constraint.end_date
    if target is None:
        return False
    if target.isoformat() in text:
        return True
    months = (
        ("janvier", "january"), ("fevrier", "february"), ("mars", "march"),
        ("avril", "april"), ("mai", "may"), ("juin", "june"),
        ("juillet", "july"), ("aout", "august"), ("septembre", "september"),
        ("octobre", "october"), ("novembre", "november"), ("decembre", "december"),
    )
    names = "|".join(months[target.month - 1])
    return bool(re.search(rf"\b0?{target.day}(?:er)?\s+(?:{names})\b", text))


def _verify_constraint_provenance(
    *, kind: str, value: str | TemporalConstraint, llm_provenance: ConstraintProvenance, user_message: str,
) -> tuple[ConstraintProvenance, str]:
    """Require an independently observable restriction relation before filtering."""
    if llm_provenance != "explicit":
        return "inferred", "LLM marked this constraint inferred"
    text = _normalized_text(user_message)
    if kind == "source_type":
        return ("explicit", "explicit document-source scope found in user request") if _has_explicit_source_scope(text, str(value)) else ("inferred", "no explicit document-source scope found in user request")
    if kind == "metadata.sender":
        pattern = _value_pattern(str(value))
        relation = rf"\b(?:envoy\w*|expediteur\w*|sender|sent)\b(?:\s+\w+){{0,4}}\s+\b(?:par|by|from)\s+{pattern}"
        return ("explicit", "explicit sender relation found in user request") if re.search(relation, text) else ("inferred", "no explicit sender restriction found in user request")
    if kind == "metadata.recipient":
        pattern = _value_pattern(str(value))
        relation = rf"\b(?:rec\w*|destinataire\w*|recipient|received)\b(?:\s+\w+){{0,4}}\s+\b(?:par|by|pour|to)\s+{pattern}"
        return ("explicit", "explicit recipient relation found in user request") if re.search(relation, text) else ("inferred", "no explicit recipient restriction found in user request")
    if kind == "metadata.thread":
        has_thread = bool(re.search(r"\b(?:fil|thread|conversation)\b", text))
        return ("explicit", "explicit thread scope found in user request") if has_thread else ("inferred", "no explicit thread scope found in user request")
    if kind == "metadata.document_type":
        source = str(value).lower()
        if source in {"email", "pdf", "file"}:
            return _verify_constraint_provenance(kind="source_type", value=source, llm_provenance=llm_provenance, user_message=user_message)
        return "inferred", "document type is not a supported explicit source scope"
    if kind == "temporal":
        constraint = value
        if not isinstance(constraint, TemporalConstraint):
            return "inferred", "invalid temporal constraint"
        if constraint.mode == "recent":
            return "inferred", "recent is always a soft preference"
        operators = {
            "before": r"\b(?:avant|before|jusqu(?:a|au)|until)\b",
            "after": r"\b(?:depuis|apres|after|since)\b",
            "range": r"\b(?:entre|between|du|from)\b",
        }
        relation_ok = constraint.mode == "exact" or bool(re.search(operators.get(constraint.mode, r"$^"), text))
        return ("explicit", "explicit temporal restriction found in user request") if relation_ok and _date_is_mentioned(text, constraint) else ("inferred", "no explicit temporal restriction found in user request")
    return "inferred", "unsupported constraint kind"


def sanitize_plan_for_retrieval(raw_plan: OrchestrationPlan, *, user_message: str = "") -> SanitizedPlan:
    """Keep only user-explicit restrictions as hard filters.

    The planner may infer useful context, but an inference must never eliminate
    a retrieved candidate. Soft preferences are deliberately not applied here:
    the stable RAG ranking remains unchanged until a future, measured boost is
    introduced.
    """
    hard_filters: list[dict[str, Any]] = []
    soft_preferences: list[dict[str, Any]] = []
    hard_sources: list[SourceType] = []
    for source in raw_plan.source_types:
        llm_provenance = raw_plan.source_type_provenance.get(source, "inferred")
        provenance, reason = _verify_constraint_provenance(kind="source_type", value=source, llm_provenance=llm_provenance, user_message=user_message)
        item = {"kind": "source_type", "value": source, "llm_provenance": llm_provenance, "validated_provenance": provenance, "validation_reason": reason}
        if provenance == "explicit":
            hard_sources.append(source)
            hard_filters.append(item)
        else:
            soft_preferences.append(item)

    hard_temporal: list[TemporalConstraint] = []
    soft_temporal: list[TemporalConstraint] = []
    for constraint in raw_plan.temporal_constraints:
        provenance, reason = _verify_constraint_provenance(kind="temporal", value=constraint, llm_provenance=constraint.provenance, user_message=user_message)
        item = {"kind": "temporal", "value": constraint.model_dump(mode="json"), "llm_provenance": constraint.provenance, "validated_provenance": provenance, "validation_reason": reason}
        # "recent" orders known candidates by date; it is never a date-range
        # exclusion, even when the user explicitly asks for recent material.
        if constraint.mode == "recent":
            soft_temporal.append(constraint)
            soft_preferences.append(item)
        elif provenance == "explicit":
            hard_temporal.append(constraint)
            hard_filters.append(item)
        else:
            soft_preferences.append(item)

    hard_metadata: dict[str, MetadataConstraint] = {}
    for field_name in ("sender", "recipient", "thread", "document_type"):
        constraint = getattr(raw_plan.metadata_constraints, field_name)
        if not constraint:
            continue
        kind = f"metadata.{field_name}"
        provenance, reason = _verify_constraint_provenance(kind=kind, value=constraint.value, llm_provenance=constraint.provenance, user_message=user_message)
        item = {"kind": kind, "value": constraint.value, "llm_provenance": constraint.provenance, "validated_provenance": provenance, "validation_reason": reason}
        if provenance == "explicit":
            hard_metadata[field_name] = constraint
            hard_filters.append(item)
        else:
            soft_preferences.append(item)

    return SanitizedPlan(
        plan=raw_plan.model_copy(update={
            "source_types": hard_sources,
            "temporal_constraints": [*hard_temporal, *soft_temporal],
            "metadata_constraints": MetadataConstraints(**hard_metadata),
        }),
        hard_filters=hard_filters,
        soft_preferences=soft_preferences,
        removed_constraints=[],
    )


def _metadata_value(meta: dict[str, Any], key: str) -> str:
    document = meta.get("document_metadata") or {}
    value = document.get(key) or (meta.get("source_metadata") or {}).get(key) or ""
    if isinstance(value, list):
        value = " ".join(map(str, value))
    return str(value).lower()


def _parse_meta_date(meta: dict[str, Any]) -> CalendarDate | None:
    value = _metadata_value(meta, "chronological_key") or _metadata_value(meta, "date")
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return CalendarDate.fromisoformat(value[:10])
        except ValueError:
            return None


def filter_retrieval_candidates(candidates: list[tuple[float, dict[str, Any]]], plan: OrchestrationPlan) -> list[tuple[float, dict[str, Any]]]:
    """Deterministic metadata filter over existing retrieved candidates only."""
    if not plan.needs_retrieval:
        return []
    result: list[tuple[float, dict[str, Any]]] = []
    for score, meta in candidates:
        if plan.source_types and str(meta.get("source", "")).lower() not in plan.source_types:
            continue
        constraints = plan.metadata_constraints
        if constraints.sender and constraints.sender.value.lower() not in _metadata_value(meta, "sender"):
            continue
        if constraints.recipient and constraints.recipient.value.lower() not in _metadata_value(meta, "recipients"):
            continue
        if constraints.thread and constraints.thread.value.lower() not in _metadata_value(meta, "thread_id"):
            continue
        if constraints.document_type and constraints.document_type.value.lower() != str(meta.get("source", "")).lower():
            continue
        candidate_date = _parse_meta_date(meta)
        matched = True
        for temporal in plan.temporal_constraints:
            if candidate_date is None:
                matched = False
            elif temporal.mode == "exact":
                matched = candidate_date == temporal.date
            elif temporal.mode == "before":
                matched = candidate_date < temporal.date
            elif temporal.mode == "after":
                matched = candidate_date > temporal.date
            elif temporal.mode == "range":
                matched = temporal.start_date <= candidate_date <= temporal.end_date
            # "recent" deliberately preserves thematic ranking; it is an ordering hint, not a date guess.
            if not matched:
                break
        if matched:
            result.append((score, meta))
    if any(item.mode == "recent" for item in plan.temporal_constraints):
        result.sort(key=lambda item: (_parse_meta_date(item[1]) or CalendarDate.min, float(item[0])), reverse=True)
    return result


def explain_candidate_rejection(meta: dict[str, Any], plan: OrchestrationPlan) -> dict[str, Any] | None:
    """Diagnostic-only mirror of the active constraint predicates; no filtering side effect."""
    if plan.source_types and str(meta.get("source", "")).lower() not in plan.source_types:
        return {"rejected_by": "source_types", "expected": plan.source_types, "actual": meta.get("source")}
    constraints = plan.metadata_constraints
    for field, metadata_key, expected in (("sender", "sender", constraints.sender), ("recipient", "recipients", constraints.recipient), ("thread", "thread_id", constraints.thread)):
        actual = _metadata_value(meta, metadata_key)
        if expected and expected.value.lower() not in actual:
            return {"rejected_by": f"metadata.{field}", "expected": expected.value, "actual": actual}
    if constraints.document_type and constraints.document_type.value.lower() != str(meta.get("source", "")).lower():
        return {"rejected_by": "metadata.document_type", "expected": constraints.document_type.value, "actual": meta.get("source")}
    candidate_date = _parse_meta_date(meta)
    for temporal in plan.temporal_constraints:
        if candidate_date is None:
            return {"rejected_by": "temporal_constraint", "reason": "candidate has no structured date"}
        if temporal.mode == "exact" and candidate_date != temporal.date:
            return {"rejected_by": "temporal_constraint", "expected": temporal.model_dump(mode="json"), "actual": candidate_date.isoformat()}
        if temporal.mode == "before" and candidate_date >= temporal.date:
            return {"rejected_by": "temporal_constraint", "expected": temporal.model_dump(mode="json"), "actual": candidate_date.isoformat()}
        if temporal.mode == "after" and candidate_date <= temporal.date:
            return {"rejected_by": "temporal_constraint", "expected": temporal.model_dump(mode="json"), "actual": candidate_date.isoformat()}
        if temporal.mode == "range" and not (temporal.start_date <= candidate_date <= temporal.end_date):
            return {"rejected_by": "temporal_constraint", "expected": temporal.model_dump(mode="json"), "actual": candidate_date.isoformat()}
    return None
