import sys
import types
from pathlib import Path

_BACK_ROOT = Path(__file__).resolve().parents[2]
if str(_BACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACK_ROOT))
if "api" not in sys.modules:
    package = types.ModuleType("api")
    package.__path__ = [str(_BACK_ROOT / "api")]
    sys.modules["api"] = package

from api.orchestration import OrchestrationPlan, explain_candidate_rejection
from api.response_trace import ResponseTraceStore


def test_trace_is_associated_with_request_and_has_bounded_memory():
    store = ResponseTraceStore(limit=2)
    for request_id in ("one", "two", "three"):
        store.start(request_id, {"original_user_message": request_id})
        store.stage(request_id, "orchestration", {"raw_plan": {"needs_retrieval": True}})
        store.finish(request_id, {"final_answer": "answer"})
    assert [item["request_id"] for item in store.list()] == ["three", "two"]
    assert store.get("one") is None
    assert store.get("three")["stages"]["answer"]["final_answer"] == "answer"


def test_rejection_explanation_reports_exact_metadata_constraint():
    plan = OrchestrationPlan(intent="document_question", needs_retrieval=True, retrieval_query="subject", response_strategy="answer", metadata_constraints={"sender": "expected"})
    reason = explain_candidate_rejection({"source": "email", "document_metadata": {"sender": "actual"}}, plan)
    assert reason == {"rejected_by": "metadata.sender", "expected": "expected", "actual": "actual"}
