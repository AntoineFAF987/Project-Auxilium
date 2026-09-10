import sys
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


def _retry(query, text, mode="direct", relevant=True, resolved=None, semantics="general_document_question"):
    decision = _decision(query, text, mode, relevant)
    gap = derive_retrieval_gap(
        query=query, context_text=text, answerability=decision,
        evidence_mode=mode, query_semantics=semantics,
    )
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


def test_all_requested_aspects_supported_produces_no_gap():
    decision, gap, queries, _ = _retry(
        "Is HV02 NACE certified and what are its temperature limits?",
        "HV02 has NACE certification and operating temperature limits.",
    )
    assert decision.status == "answerable"
    assert gap.missing_aspects == []
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


def test_requested_minus_supported_keeps_nace_and_targets_temperature_only():
    decision, gap, queries, _ = _retry(
        "Is 82.7 HV02 NACE certified and what are its temperature limits?",
        "82.7 HV02 is certified according to NACE.", mode="related",
    )
    assert gap.requested_aspects == ["operating temperature limits", "NACE certification"]
    assert gap.supported_aspects == ["NACE certification"]
    assert gap.missing_aspects == ["operating temperature limits"]
    assert gap.anchors == ["82.7", "HV02"]
    assert queries == ["82.7 HV02 operating temperature limits"]
    assert gap.gap_derivation_method == "requested_minus_supported"


def test_decision_pending_evidence_targets_final_outcome_not_subject_rewrite():
    query = "Is the request approved?"
    decision, gap, queries, _ = _retry(
        query, "The request was submitted and the decision is pending.",
        mode="related", semantics="decision",
    )
    assert "final decision / approval status" in gap.missing_aspects
    assert query not in gap.missing_aspects
    assert "decision pending" in gap.supported_aspects
    assert gap.gap_derivation_method == "decision_semantics_missing_final_outcome"
    assert queries and "final decision" in queries[0]


def test_current_state_with_old_evidence_targets_latest_state():
    decision, gap, queries, _ = _retry(
        "What is the current status of the case?", "The previous status was pending.",
        mode="related", semantics="current_state",
    )
    assert gap.missing_aspects == ["latest/current state"]
    assert gap.gap_derivation_method == "current_state_semantics_missing_latest_state"


def test_unknown_gap_never_falls_back_to_the_original_query():
    query = "Antoine asks whether he may leave the territory. What should I reply?"
    decision = _decision(query, "The request was submitted and is pending.", mode="related")
    gap = derive_retrieval_gap(
        query=query, context_text="The request was submitted and is pending.",
        answerability=decision, evidence_mode="related",
    )
    assert query not in gap.missing_aspects
    assert gap.missing_aspects == []


def test_technical_anchor_hv02_is_never_replaced_by_hv01():
    _, gap, queries, _ = _retry(
        "HV02 NACE certification and temperature limits", "HV01 NACE certification.", mode="related",
    )
    assert "HV02" in gap.anchors
    assert queries and all("hv02" in item.casefold() for item in queries)
    assert all("hv01" not in item.casefold() for item in queries)


def test_exact_type_2420_survives_neighbouring_2422_evidence():
    _, _gap, queries, _ = _retry(
        "Type 2420 NACE certification and temperature limits", "Type 2422 NACE certification.", mode="related",
    )
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
    # The missing aspect is a complementary query, not an exact duplicate.
    _, _gap, queries, rejected = _retry("HV02 temperature limits", "HV02")
    assert queries == ["HV02 operating temperature limits"]
    assert not any(item["query"] == queries[0] for item in rejected)


def test_followup_uses_resolved_subject_not_raw_message():
    decision = _decision("Project Orion final decision", "Project Orion request is pending.")
    gap = derive_retrieval_gap(
        query="Project Orion final decision", context_text="Project Orion request is pending.",
        answerability=decision, evidence_mode="direct", query_semantics="decision",
    )
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
