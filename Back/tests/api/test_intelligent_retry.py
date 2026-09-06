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

from api.intelligent_retry import (  # noqa: E402
    build_retry_queries, derive_retrieval_gap, merge_cumulative_evidence,
)
from rag_core.context_sufficiency import evaluate_answerability  # noqa: E402


def _decision(query, text, mode="direct", relevant=True):
    return evaluate_answerability(
        query=query, evidence_mode=mode, context_is_relevant=relevant,
        evidence_sufficient=True, reranker_accepted=True,
        blocks=[{"text": text, "document_id": "doc"}] if text else [],
    )


def _retry(query, text, mode="direct", relevant=True, resolved=None):
    decision = _decision(query, text, mode, relevant)
    gap = derive_retrieval_gap(query=query, context_text=text, answerability=decision, evidence_mode=mode)
    queries, rejected = build_retry_queries(
        original_user_query=query, orchestrator_query=query,
        resolved_retrieval_query=resolved or query, gap=gap,
    )
    return decision, gap, queries, rejected


def test_answerable_question_does_not_need_retry_gap():
    decision, gap, queries, _ = _retry("Model HV02 NACE certification", "Model HV02 NACE certification is confirmed.")
    assert decision.status == "answerable"
    # Pipeline only invokes build_retry_queries for partial/unanswerable.
    assert not gap.missing_aspects
    assert queries == []


def test_partial_question_targets_only_missing_aspect_and_keeps_initial_evidence():
    query = "Is HV02 NACE certified and what are its temperature limits?"
    decision, gap, queries, _ = _retry(query, "HV02 is NACE certified.", mode="related")
    assert decision.status == "partial"
    assert any("temperature" in item.casefold() for item in gap.missing_aspects)
    assert all("nace certified" not in item.casefold() for item in queries)
    initial = [(0.9, {"chunk_uid": "nace", "document_id": "a", "chunk_id": 1, "text": "NACE"})]
    retry = [(0.8, {"chunk_uid": "temp", "document_id": "a", "chunk_id": 2, "text": "temperature"})]
    merged = merge_cumulative_evidence(initial, retry, retry_query_count=1)
    assert {meta["chunk_uid"] for _, meta in merged} == {"nace", "temp"}


def test_technical_anchor_hv02_is_never_replaced_by_hv01():
    _, gap, queries, _ = _retry("HV02 temperature limits", "HV01 temperature limits", mode="related")
    assert "HV02" in gap.anchors
    assert queries and all("hv02" in item.casefold() for item in queries)
    assert all("hv01" not in item.casefold() for item in queries)


def test_exact_type_2420_survives_neighbouring_2422_evidence():
    _, _gap, queries, _ = _retry("What are type 2420 technical limits?", "Type 2422 technical limits", mode="related")
    assert queries and all("2420" in item for item in queries)
    assert all("2422" not in item for item in queries)


def test_email_decision_retry_merges_old_and_new_evidence_with_provenance():
    initial = [(0.9, {"chunk_uid": "old-header", "document_id": "thread", "chunk_id": 1, "text": "response pending", "retrieved_by": ["initial"]})]
    retry = [(0.8, {"chunk_uid": "new-body", "document_id": "thread", "chunk_id": 2, "text": "final decision approved", "retrieved_by": ["retry_1"]})]
    merged = {meta["chunk_uid"]: meta for _, meta in merge_cumulative_evidence(initial, retry, retry_query_count=1)}
    assert set(merged) == {"old-header", "new-body"}
    assert "initial" in merged["old-header"]["retrieved_by"]
    assert "retry_1" in merged["new-body"]["retrieved_by"]


def test_redundant_retry_query_is_rejected():
    # A one-facet query yields the original wording; it must not relaunch it.
    _, _gap, queries, rejected = _retry("HV02 temperature limits", "HV02")
    assert queries == []
    assert rejected and rejected[0]["reason"] == "near_duplicate_of_initial_or_retry"


def test_followup_uses_resolved_subject_not_raw_message():
    decision = _decision("Project Orion final decision", "Project Orion request is pending.")
    gap = derive_retrieval_gap(query="Project Orion final decision", context_text="Project Orion request is pending.", answerability=decision, evidence_mode="direct")
    queries, _ = build_retry_queries(
        original_user_query="cherche encore", orchestrator_query="Project Orion final decision",
        resolved_retrieval_query="Project Orion final decision", gap=gap,
    )
    assert queries and all("project orion" in item.casefold() for item in queries)
    assert all("cherche encore" not in item.casefold() for item in queries)


def test_no_new_evidence_does_not_create_another_retry_round():
    decision, gap, queries, _ = _retry("HV02 temperature limits", "HV01 temperature limits", mode="related")
    assert decision.status in {"partial", "unanswerable"}
    assert len(queries) <= 2
    # The retry API has no recursive control flow: caller receives one batch.
    assert gap.answerability_before_retry == decision.status
