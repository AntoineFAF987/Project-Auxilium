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
