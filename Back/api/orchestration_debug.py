"""In-memory observability for the one-shot orchestrator; never persisted."""
from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock
from typing import Any

from .orchestration import OrchestrationPlan

_LOCK = Lock()
_LAST_SNAPSHOT: dict[str, Any] | None = None


def record_snapshot(plan: OrchestrationPlan | None, *, orchestration_ms: float, fallback_used: bool) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "orchestration_ms": orchestration_ms,
        "fallback_used": fallback_used,
    }
    if plan:
        snapshot.update(plan.model_dump(mode="json"))
    with _LOCK:
        global _LAST_SNAPSHOT
        _LAST_SNAPSHOT = snapshot
    return snapshot


def get_last_snapshot() -> dict[str, Any] | None:
    with _LOCK:
        return dict(_LAST_SNAPSHOT) if _LAST_SNAPSHOT else None
