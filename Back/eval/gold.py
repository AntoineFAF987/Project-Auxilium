"""Small, deterministic evaluation helpers for the manually curated gold suite.

This module is evaluation-only.  It deliberately imports no production
orchestration code; the optional live runner owns that boundary.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class GoldCase(BaseModel):
    """One manually maintainable gold case (one JSON object per JSONL line)."""

    id: str
    category: str
    question: str
    origin_type: str
    origin_reference: str
    why_this_case_matters: str
    expected_source_types: list[str] = Field(default_factory=list)
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_chunk_uids: list[str] = Field(default_factory=list)
    answerable: bool
    required_facts: list[list[str]] = Field(default_factory=list)
    optional_facts: list[list[str]] = Field(default_factory=list)
    forbidden_facts: list[list[str]] = Field(default_factory=list)
    exact_entities: list[str] = Field(default_factory=list)
    query_semantics: str
    source_constraints: dict[str, list[str]] = Field(default_factory=lambda: {"hard": [], "soft": []})
    conversation_context: list[dict[str, str]] | None = None
    expected_behavior: dict[str, bool]
    execution_scope: str = "controlled"

    @model_validator(mode="after")
    def _validate(self):
        if self.origin_type not in {"test", "real_trace", "corpus_document"}:
            raise ValueError("origin_type must be test, real_trace, or corpus_document")
        if self.execution_scope not in {"indexed_local", "controlled", "live_only"}:
            raise ValueError("unknown execution_scope")
        if not self.origin_reference or not self.why_this_case_matters:
            raise ValueError("gold provenance and rationale are required")
        if not self.answerable and self.expected_document_ids:
            raise ValueError("unanswerable cases must not claim an expected document")
        return self


def load_gold_cases(path: str | Path) -> list[GoldCase]:
    import json

    cases: list[GoldCase] = []
    ids: set[str] = set()
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            case = GoldCase.model_validate(json.loads(raw))
        except Exception as exc:
            raise ValueError(f"invalid gold case at line {number}: {exc}") from exc
        if case.id in ids:
            raise ValueError(f"duplicate gold id: {case.id}")
        ids.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError("gold suite is empty")
    return cases


def _matches(text: str, alternatives: list[str]) -> bool:
    """All terms in one alternative group must match, accent/case insensitive."""
    normalized = text.casefold()
    return all(re.search(term.casefold(), normalized, flags=re.IGNORECASE) for term in alternatives)


def score_answer(case: GoldCase, answer: str, *, sources: list[dict[str, Any]] | None = None,
                 abstained: bool = False, latency_ms: float | None = None,
                 retrieval_trace: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rule-only answer scoring; each fact accepts several regex formulations."""
    required = [_matches(answer, group) for group in case.required_facts]
    forbidden = [_matches(answer, group) for group in case.forbidden_facts]
    returned_types = {str(item.get("source") or item.get("source_type") or "") for item in (sources or [])}
    hard = set(case.source_constraints.get("hard", []))
    exact_ok = not case.exact_entities or all(entity.casefold() in answer.casefold() for entity in case.exact_entities)
    return {
        "required_fact_recall": sum(required) / len(required) if required else 1.0,
        "forbidden_claim_count": sum(forbidden),
        "answerability_correct": abstained is (not case.answerable),
        "source_constraint_correct": not hard or hard.issubset(returned_types),
        "exact_entity_correct": exact_ok,
        "current_state_correct": (
            sum(required) / len(required) if case.query_semantics in {"current_state", "decision"} and required else None
        ),
        "citation_presence": bool(re.search(r"(?:\[\d+\]|<CITATIONS>)", answer)),
        "answer_length": len(answer),
        "latency_ms": latency_ms,
        "retry_success": retry_success(case, retrieval_trace, answer),
    }


def retry_success(case: GoldCase, trace: dict[str, Any] | None, final_answer: str = "") -> float | None:
    """A retry succeeds only when a retry round occurred and closes a required gap.

    Traces differ slightly between pipeline versions, hence the two supported
    keys. ``None`` means this execution mode did not expose a retry trace.
    """
    if not case.expected_behavior.get("should_retry"):
        return None
    trace = trace or {}
    rounds = trace.get("retry_rounds") or trace.get("intelligent_retry_rounds") or []
    if not rounds:
        return 0.0
    return float(all(_matches(final_answer, group) for group in case.required_facts))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)

    def one(items: list[dict[str, Any]]) -> dict[str, float | int | None]:
        metric_names = ("document_hit", "context_recall", "exact_entity_violation", "retry_success",
                        "required_fact_recall", "forbidden_claim_count", "answerability_correct", "current_state_correct",
                        "source_constraint_correct", "exact_entity_correct", "citation_presence", "latency_ms")
        result: dict[str, float | int | None] = {"case_count": len(items)}
        for name in metric_names:
            values = [item[name] for item in items if item.get(name) is not None]
            result[name] = sum(values) / len(values) if values else None
        return result

    return {"global": one(rows), "by_category": {key: one(value) for key, value in sorted(grouped.items())}}
