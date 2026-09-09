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
    GroundedConversationContext, decide_conversation_retrieval, decide_next_source_action, derive_active_source_context,
    derive_grounded_conversation_context, evaluate_grounding_candidate,
    execute_source_plan, explicit_source_constraint, match_structured_values,
    normalize_source_plan, structured_clarification,
)


def _grounded(**overrides):
    value = dict(
        active_subject="validated subject", primary_document_ids=("doc-1",), active_chunk_uids=("doc-1:0",),
        active_source_type="email", active_email_document_id="doc-1", active_email_message_id="message-1",
        active_attachment_document_ids=("attachment-1",), supported_claims=("approved with conditions",),
        grounding_valid=True,
    )
    value.update(overrides)
    return GroundedConversationContext(**value)


def test_scope_uses_direct_attachment_relation_before_search():
    decision = decide_conversation_retrieval("Que dit la pièce jointe ?", _grounded(), reuse_previous_subject=True)
    assert decision.retrieval_scope == "structural_relations"
    assert decision.requires_structural_lookup


def test_precise_name_of_joined_object_remains_a_structural_followup():
    decision = decide_conversation_retrieval(
        "Que dit l'ordonnance jointe au mail du 28 août ?", _grounded(),
        intent="document_question", reuse_previous_subject=False,
    )
    assert decision.retrieval_scope == "structural_relations"
    assert decision.reason == "direct_relation_of_active_document"


def test_unresolved_attachment_is_a_strong_continuity_signal():
    decision = decide_conversation_retrieval(
        "Et son contenu ?", _grounded(unresolved_information_needs=("attachment_content",)),
        intent="document_question", reuse_previous_subject=False,
    )
    assert decision.retrieval_scope == "structural_relations"
    assert decision.reason == "unresolved_structural_need_matches_follow_up"


def test_pdf_reference_uses_active_document_without_planner_reuse_flag():
    decision = decide_conversation_retrieval("Et le PDF ?", _grounded(), intent="document_question")
    assert decision.retrieval_scope == "active_documents"


def test_referential_unknown_attachment_name_uses_unique_relation():
    decision = decide_conversation_retrieval("Trouve-moi cette ordonnance", _grounded(), intent="document_question")
    assert decision.retrieval_scope == "structural_relations"


def test_scope_searches_only_active_document_for_referential_detail():
    decision = decide_conversation_retrieval("Quelle est sa température maximale ?", _grounded(active_source_type="local"), reuse_previous_subject=True)
    assert decision.retrieval_scope == "active_documents"
    assert decision.reuse_active_documents


def test_scope_reuses_supported_grounding_for_transform_or_conclusion():
    decision = decide_conversation_retrieval("Donc, en gros, c'est autorisé ?", _grounded(), intent="conversation", needs_retrieval=False)
    assert decision.retrieval_scope == "none"
    assert decision.reuse_prior_grounding


def test_scope_expands_within_active_source_for_freshness():
    decision = decide_conversation_retrieval("Y a-t-il un mail plus récent ?", _grounded(), reuse_previous_subject=True)
    assert decision.retrieval_scope == "active_source"


def test_scope_uses_global_for_incompatible_new_subject():
    decision = decide_conversation_retrieval("Quelle est la bande morte du système ?", _grounded(), intent="document_question", reuse_previous_subject=False)
    assert decision.retrieval_scope == "global"


def test_last_successful_grounding_survives_failed_followup():
    history = [
        {"role": "assistant", "content": "grounded", "meta": {"evidence_provenance": {"evidence_sufficient": True, "evidence_chunk_uids": ["a:1"], "evidence_document_ids": ["a"]}}},
        {"role": "assistant", "content": "I cannot answer", "meta": {"evidence_provenance": {"evidence_sufficient": False}}},
    ]
    context = derive_grounded_conversation_context(history)
    assert context.grounding_valid and context.primary_document_ids == ("a",)
    assert context.source_turn == 0


def test_clarification_with_stale_provenance_cannot_replace_grounding():
    history = [
        {"role": "assistant", "content": "good", "meta": {"mode": "STRICT(local)", "evidence_provenance": {"evidence_sufficient": True, "evidence_chunk_uids": ["a:1"], "evidence_document_ids": ["a"]}}},
        {"role": "assistant", "content": "Which one?", "meta": {"mode": "CLARIFICATION", "evidence_provenance": {"evidence_sufficient": True, "evidence_chunk_uids": ["noise:1"], "evidence_document_ids": ["noise"]}}},
    ]
    context = derive_grounded_conversation_context(history)
    assert context.primary_document_ids == ("a",)
    assert context.source_turn == 0


def test_explicitly_rejected_candidate_cannot_replace_grounding_after_timeout():
    history = [
        {"role": "assistant", "content": "good", "meta": {"evidence_provenance": {"evidence_sufficient": True, "evidence_chunk_uids": ["a:1"], "evidence_document_ids": ["a"]}}},
        {"role": "assistant", "content": "failed", "meta": {"grounding_candidate_valid": False, "grounding_candidate_rejected_reason": "orchestrator_failed", "evidence_provenance": {"evidence_sufficient": True, "evidence_chunk_uids": ["noise:1"], "evidence_document_ids": ["noise"]}}},
    ]
    context = derive_grounded_conversation_context(history)
    assert context.primary_document_ids == ("a",)
    assert context.source_turn == 0


def _grounding_inputs(**overrides):
    validations = {
        "orchestrator_failed": True,
        "answerability": {"answerability": "answerable"},
        "evidence_provenance": {
            "evidence_sufficient": True,
            "primary_document_ids": ["doc-a"],
            "evidence_document_ids": ["doc-a"],
            "evidence_chunk_uids": ["doc-a:1"],
        },
        "claim_sources": [{"claim": "supported", "source_type": "email", "source_id": "doc-a"}],
    }
    validations.update(overrides)
    return validations


def test_orchestrator_failure_does_not_invalidate_final_documentary_evidence():
    evaluation = evaluate_grounding_candidate(
        mode="STRICT(local)", status="answered", validations=_grounding_inputs(),
        sources=[{"document_id": "doc-a"}],
    )
    assert evaluation == {"valid": True, "rejected_reason": None, "basis": "final_evidence_quality"}


def test_first_fallback_grounding_is_reused_for_structural_followup():
    history = [
        {"role": "user", "content": "What is in the email?"},
        {
            "role": "assistant", "content": "The email says X.",
            "meta": {
                "grounding_candidate_valid": True,
                "grounding_validation_basis": "final_evidence_quality",
                "grounding_origin": "fallback",
                "orchestrator_failed": True,
                "orchestrator_failed_but_grounding_recovered": True,
                "answerability": {"answerability": "answerable"},
                "evidence_provenance": {
                    "evidence_sufficient": True, "evidence_chunk_uids": ["email-a:1"],
                    "evidence_document_ids": ["email-a"], "primary_document_ids": ["email-a"],
                },
                "sources": [{"document_id": "email-a", "path": "email-a.eml", "source": "email"}],
                "email_target_document_id": "email-a", "email_target_message_id": "message-a",
                "attachment_states": [{"attachment_document_id": "attachment-a"}],
                "unresolved_information_needs": ["attachment_content"],
            },
        },
    ]
    context = derive_grounded_conversation_context(history)
    assert context.grounding_valid and context.source_turn == 1
    decision = decide_conversation_retrieval("What does the attachment say?", context, intent="document_question")
    assert decision.retrieval_scope == "structural_relations"
    assert decision.requires_global_search is False


def test_healthy_orchestrator_does_not_validate_insufficient_evidence():
    evaluation = evaluate_grounding_candidate(
        mode="STRICT(local)", status="answered",
        validations=_grounding_inputs(orchestrator_failed=False, evidence_provenance={"evidence_sufficient": False}),
        sources=[{"document_id": "doc-a"}],
    )
    assert evaluation["valid"] is False
    assert evaluation["rejected_reason"] == "evidence_not_sufficient"


def test_clarification_and_abstention_are_not_grounding_candidates():
    for mode, status in (("CLARIFICATION", "answered"), ("ABSTAIN", "abstained")):
        evaluation = evaluate_grounding_candidate(
            mode=mode, status=status, validations=_grounding_inputs(), sources=[{"document_id": "doc-a"}],
        )
        assert evaluation["valid"] is False


def test_partial_documentary_grounding_can_preserve_email_and_attachment_context():
    history = [{
        "role": "assistant", "content": "Attachment identified; its content is unavailable.",
        "meta": {
            "grounding_candidate_valid": True,
            "grounding_origin": "fallback",
            "orchestrator_failed": True,
            "answerability": {"answerability": "partial"},
            "evidence_provenance": {
                "evidence_sufficient": True, "evidence_chunk_uids": ["email-a:1"],
                "evidence_document_ids": ["email-a"], "primary_document_ids": ["email-a"],
            },
            "sources": [{"document_id": "email-a", "path": "email-a.eml", "source": "email"}],
            "email_target_document_id": "email-a", "email_target_message_id": "message-a",
            "attachment_states": [{"attachment_document_id": "attachment-a"}],
            "unresolved_information_needs": ["attachment_content"],
        },
    }]
    context = derive_grounded_conversation_context(history)
    assert context.grounding_valid
    assert context.primary_document_ids == ("email-a",)
    assert context.active_attachment_document_ids == ("attachment-a",)
    assert context.unresolved_information_needs == ("attachment_content",)
    decision = decide_conversation_retrieval("Que dit la pièce jointe ?", context, intent="document_question")
    assert decision.retrieval_scope == "structural_relations"


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
