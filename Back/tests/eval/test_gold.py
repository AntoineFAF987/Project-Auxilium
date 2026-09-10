from pathlib import Path

from eval.gold import aggregate, load_gold_cases, score_answer


DATASET = Path(__file__).resolve().parents[2] / "eval" / "data" / "gold_v1.jsonl"


def test_gold_dataset_is_complete_provenanced_and_balanced():
    cases = load_gold_cases(DATASET)
    assert len(cases) == 50
    assert {case.category for case in cases} == {
        "factual_direct", "exact_entity", "current_state_decision", "email_attachment",
        "source_constraints", "followup_transformation", "multilingual", "structured_xlsx",
        "answerability_related", "intelligent_retry",
    }
    assert all(case.origin_reference and case.why_this_case_matters for case in cases)
    assert all(case.origin_type in {"test", "real_trace", "corpus_document"} for case in cases)


def test_answer_scoring_is_pattern_based_and_deterministic():
    case = next(case for case in load_gold_cases(DATASET) if case.id == "fd-api-limits")
    score = score_answer(case, "Free : 100 ; Pro : 500 ; Enterprise est illimitée. [1]", sources=[{"source": "email"}])
    assert score["required_fact_recall"] == 1.0
    assert score["forbidden_claim_count"] == 0
    assert score["citation_presence"] is True


def test_aggregate_ignores_controlled_skips_for_numeric_averages():
    report = aggregate([
        {"category": "a", "document_hit": 1.0, "context_recall": 1.0, "exact_entity_violation": 0.0,
         "retry_success": None, "required_fact_recall": None, "forbidden_claim_count": None,
         "answerability_correct": None, "current_state_correct": None, "source_constraint_correct": None,
         "exact_entity_correct": None, "citation_presence": None, "latency_ms": 10.0},
        {"category": "a", "document_hit": None, "context_recall": None, "exact_entity_violation": None,
         "retry_success": None, "required_fact_recall": None, "forbidden_claim_count": None,
         "answerability_correct": None, "current_state_correct": None, "source_constraint_correct": None,
         "exact_entity_correct": None, "citation_presence": None, "latency_ms": None},
    ])
    assert report["global"]["document_hit"] == 1.0
    assert report["global"]["case_count"] == 2


def test_retry_success_requires_a_retry_round_and_closed_required_gap():
    case = next(case for case in load_gold_cases(DATASET) if case.id == "retry-partial-hv02")
    assert score_answer(case, "HV02 has NACE certification and temperature limits.", retrieval_trace={})["retry_success"] == 0.0
    assert score_answer(case, "HV02 has NACE certification and temperature limits.", retrieval_trace={"retry_rounds": [{}]})["retry_success"] == 1.0
