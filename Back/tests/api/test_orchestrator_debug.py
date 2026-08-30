import sys
import types
from pathlib import Path
from unittest.mock import patch

_BACK_ROOT = Path(__file__).resolve().parents[2]
if str(_BACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACK_ROOT))
if "api" not in sys.modules:
    package = types.ModuleType("api")
    package.__path__ = [str(_BACK_ROOT / "api")]
    sys.modules["api"] = package

from api.orchestration import OrchestrationPlan
from api.routes_orchestrator_debug import (
    DebugHistoryMessage,
    DebugPlanRequest,
    get_orchestrator_debug_config,
    get_orchestrator_last_plan,
    test_orchestrator_plan as run_debug_plan,
)


def _plan():
    return OrchestrationPlan(
        intent="refine_previous_search", needs_retrieval=True,
        retrieval_query="standalone active subject", use_history=True,
        reuse_previous_subject=True, source_types=["email"], response_strategy="answer",
    )


def test_debug_config_exposes_effective_prompt_without_secrets():
    details = {
        "enabled": False, "provider": "mistral", "model": "planner-model", "timeout": 8,
        "system_prompt": "actual runtime prompt",
    }
    with patch("api.routes_orchestrator_debug._runtime_details", return_value=details):
        response = get_orchestrator_debug_config()
    assert response == details
    assert "key" not in response and "token" not in response and "secret" not in response


def test_debug_plan_executes_only_the_orchestrator_and_records_snapshot():
    request = DebugPlanRequest(
        message="Search emails instead",
        history=[DebugHistoryMessage(role="user", content="Active subject")],
    )
    with patch("api.routes_orchestrator_debug._execute_plan", return_value=_plan()) as execute:
        response = run_debug_plan(request)
    execute.assert_called_once_with("Search emails instead", [{"role": "user", "content": "Active subject"}])
    assert response["fallback_used"] is False
    assert response["plan"]["source_types"] == ["email"]
    assert get_orchestrator_last_plan()["snapshot"]["intent"] == "refine_previous_search"


def test_debug_plan_exposes_fallback_without_retrieval_or_persistence():
    request = DebugPlanRequest(message="Test")
    with patch("api.routes_orchestrator_debug._execute_plan", side_effect=ValueError("bad output")) as execute:
        response = run_debug_plan(request)
    execute.assert_called_once()
    assert response["plan"] is None
    assert response["validated_plan"] is None
    assert response["fallback_used"] is True
    assert response["validation_error"] == "ValueError"
    assert get_orchestrator_last_plan()["snapshot"]["fallback_used"] is True
