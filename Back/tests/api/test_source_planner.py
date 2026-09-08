import sys
import types
from pathlib import Path

_BACK_ROOT = Path(__file__).resolve().parents[2]
if "api" not in sys.modules:
    package = types.ModuleType("api")
    package.__path__ = [str(_BACK_ROOT / "api")]
    sys.modules["api"] = package

from api.source_planner import (  # noqa: E402
    ActiveSourceContext, SourceEvidenceResult, SourcePlanItem, annotate_active_source_candidates,
    decide_next_source_action, derive_active_source_context,
    execute_source_plan, explicit_source_constraint, match_structured_values,
    normalize_source_plan, structured_clarification,
)


def test_gefa_followup_keeps_active_local_source_before_web():
    history = [{
        "role": "assistant", "content": "Oui.",
        "meta": {"sources": [{"file": "TARIF GEFA 2025.pdf", "source": "pdf"}], "generation_mode": "strict_local"},
    }]
    active = derive_active_source_context(history)
    plan = normalize_source_plan(
        [SourcePlanItem(source="web", priority=1), SourcePlanItem(source="local", priority=2)],
        mode="auto", intent="refine_previous_search", web_request_explicit=False, active=active,
    )
    assert active.label == "TARIF GEFA 2025.pdf"
    assert plan[0].source == "local"
    assert all(item.source != "web" for item in plan)


def test_active_document_is_marked_without_changing_rrf_order():
    rows = [
        (0.05, {"file": "generic.pdf"}),
        (0.02, {"file": "TARIF GEFA 2025.pdf"}),
    ]
    annotated = annotate_active_source_candidates(
        rows,
        ActiveSourceContext(source="local", source_ids=("C:/docs/TARIF GEFA 2025.pdf",)),
    )
    assert [score for score, _meta in annotated] == [0.05, 0.02]
    assert annotated[0][1]["active_source_context_match"] is False
    assert annotated[1][1]["active_source_context_match"] is True


def test_explicit_price_list_is_hard_local_constraint():
    constraint = explicit_source_constraint("Cherche dans la price list et donne-moi le prix")
    plan = normalize_source_plan(
        [SourcePlanItem(source="web", priority=1), SourcePlanItem(source="local", priority=2)],
        mode="auto", intent="document_question", web_request_explicit=False,
        active=ActiveSourceContext(), explicit_sources=constraint,
    )
    assert [item.source for item in plan] == ["local"]


def test_required_second_source_executes_but_optional_stops_when_answerable():
    calls = []
    def execute(item):
        calls.append(item.source)
        return SourceEvidenceResult(item.source, 0.9, answerability="answerable", source_priority=item.priority)
    required = execute_source_plan(
        [SourcePlanItem(source="local", priority=1, required=True), SourcePlanItem(source="general", priority=2, required=True)],
        {"local": execute, "general": execute}, max_source_expansions=2,
    )
    assert calls == ["local", "general"]
    calls.clear()
    optional = execute_source_plan(
        [SourcePlanItem(source="local", priority=1, required=True), SourcePlanItem(source="web", priority=2)],
        {"local": execute, "web": execute}, max_source_expansions=2,
    )
    assert calls == ["local"]
    assert optional.sources_skipped[0]["reason"] == "previous_source_answerable"


def test_complementary_general_source_runs_for_a_distinct_claim():
    calls = []
    def execute(item):
        calls.append(item.source)
        return SourceEvidenceResult(item.source, 0.9, answerability="answerable", source_priority=item.priority)
    execute_source_plan(
        [
            SourcePlanItem(source="local", priority=1, required=True),
            SourcePlanItem(source="general", priority=2, complementary=True),
        ],
        {"local": execute, "general": execute},
    )
    assert calls == ["local", "general"]


def test_bounded_source_expansion_and_next_action():
    plan = [SourcePlanItem(source="local", priority=1), SourcePlanItem(source="web", priority=2)]
    assert decide_next_source_action(
        answerability="unanswerable", clarification_needed=False,
        current_sources_checked=["local"], source_plan=plan, max_source_expansions=2,
    ) == "SEARCH_WEB"
    assert decide_next_source_action(
        answerability="partial", clarification_needed=False,
        current_sources_checked=["local", "web"], source_plan=plan, max_source_expansions=1,
    ) == "ANSWER_PARTIAL"


def test_structured_match_preserves_row_column_value_and_origin():
    block = {"blocks": [{
        "block_type": "table_row",
        "source_metadata": {
            "sheet_name": "Prix d'achat HT", "table_id": "prices",
            "row_key": "50", "structured_cells": [
                {"column_key": "DN", "column_name": "DN", "cell_value": "50", "value_origin": "explicit"},
                {"column_key": "KG2 2023", "column_name": "KG2 2023", "cell_value": "54.21", "value_origin": "explicit"},
            ],
        },
    }]}
    result = match_structured_values("prix KG2 DN50 2023", [block])
    assert result["structured_match"] is True
    assert result["structured_match_details"][0]["matched_row"] == "50"
    assert result["structured_match_details"][0]["matched_column"] == "KG2 2023"
    assert result["structured_match_details"][0]["matched_value"] == "54.21"
    assert result["value_origin"] == "explicit"


def test_multiple_structured_price_labels_trigger_blocking_clarification():
    match = {
        "ambiguous_value_types": True,
        "structured_match_details": [
            {"sheet_name": "Prix d'achat HT", "matched_column": "KG2 2023", "matched_value": "54.21"},
            {"sheet_name": "Prix de vente France", "matched_column": "KG2 2023", "matched_value": "82.00"},
        ],
    }
    clarification = structured_clarification("prix KG2 DN50", match)
    assert clarification is not None
    assert clarification["ambiguity_level"] == "blocking"
    assert "Prix d'achat HT" in clarification["clarification_question"]
    assert "Prix de vente France" in clarification["clarification_question"]


def test_explicit_structured_price_label_avoids_clarification():
    match = {
        "ambiguous_value_types": True,
        "structured_match_details": [
            {"sheet_name": "Prix d'achat HT", "matched_column": "KG2 2023", "matched_value": "54.21"},
            {"sheet_name": "Prix de vente France", "matched_column": "KG2 2023", "matched_value": "82.00"},
        ],
    }
    assert structured_clarification("prix d'achat HT KG2 DN50 2023", match) is None


def test_structured_formula_value_is_reported_as_computed():
    block = {"blocks": [{
        "block_type": "table_row",
        "source_metadata": {
            "sheet_name": "Prix de vente France", "table_id": "sales",
            "row_key": "50", "structured_cells": [{
                "column_key": "KG2 2023", "column_name": "KG2 2023",
                "cell_value": "82.00", "value_origin": "computed",
            }],
        },
    }]}
    result = match_structured_values("prix KG2 DN50 2023", [block])
    assert result["structured_match"] is True
    assert result["value_origin"] == "computed"
