"""Local debug endpoints for inspecting orchestration without executing RAG."""
from __future__ import annotations

import time
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .orchestration import OrchestrationPlanOutputError, SYSTEM_PROMPT, sanitize_plan_for_retrieval
from .orchestration_debug import get_last_snapshot, record_snapshot

router = APIRouter(prefix="/debug/orchestrator", tags=["debug"])


class DebugHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class DebugPlanRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[DebugHistoryMessage] = Field(default_factory=list, max_length=8)


def _runtime_details() -> dict:
    # Import lazily: the route itself must not construct/load the retrieval index.
    from .answer_pipeline import ORCHESTRATOR_SETTINGS, _RUNTIME_SETTINGS

    return {
        "enabled": ORCHESTRATOR_SETTINGS.enabled,
        "provider": _RUNTIME_SETTINGS.generation.provider,
        "model": ORCHESTRATOR_SETTINGS.model or _RUNTIME_SETTINGS.generation.model,
        "timeout_seconds": ORCHESTRATOR_SETTINGS.timeout_seconds,
        "slow_warning_seconds": ORCHESTRATOR_SETTINGS.slow_warning_seconds,
        "reasoning_effort": ORCHESTRATOR_SETTINGS.reasoning_effort,
        "max_output_tokens": ORCHESTRATOR_SETTINGS.max_output_tokens,
        "system_prompt": SYSTEM_PROMPT,
    }


def _execute_plan(message: str, history: list[dict]):
    """The exact production planner call only; no answer pipeline execution."""
    from .answer_pipeline import _run_orchestration

    return _run_orchestration(message, history)


@router.get("/config")
def get_orchestrator_debug_config():
    """Expose effective planner configuration, never environment secrets."""
    return _runtime_details()


@router.get("/last-plan")
def get_orchestrator_last_plan():
    return {"snapshot": get_last_snapshot()}


@router.post("/plan")
def test_orchestrator_plan(body: DebugPlanRequest):
    """Run exactly one plan call, with no retrieval, answer generation or persistence."""
    started = time.perf_counter()
    history = [item.model_dump() for item in body.history]
    try:
        plan = _execute_plan(body.message, history)
        sanitized = sanitize_plan_for_retrieval(plan, user_message=body.message)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        record_snapshot(plan, orchestration_ms=latency_ms, fallback_used=False)
        return {
            "plan": plan.model_dump(mode="json"),
            "validated_plan": sanitized.plan.model_dump(mode="json"),
            "constraint_decisions": {
                "hard_filters": sanitized.hard_filters,
                "soft_preferences": sanitized.soft_preferences,
                "removed": sanitized.removed_constraints,
            },
            "latency_ms": latency_ms,
            "metrics": dict(plan._orchestration_metrics),
            "fallback_used": False,
            "raw_model_output": plan._raw_model_output,
            "validation_error": None,
            "validation_error_details": None,
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        record_snapshot(None, orchestration_ms=latency_ms, fallback_used=True)
        return {
            "plan": None,
            "validated_plan": None,
            "latency_ms": latency_ms,
            "fallback_used": True,
            "validation_error": type(exc).__name__,
            "validation_error_details": exc.validation_error_details if isinstance(exc, OrchestrationPlanOutputError) else None,
            "raw_model_output": exc.raw_model_output if isinstance(exc, OrchestrationPlanOutputError) else None,
            "metrics": dict(getattr(exc, "orchestration_metrics", {}) or {}),
        }
