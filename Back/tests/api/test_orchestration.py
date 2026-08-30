import json
import sys
import types
from datetime import date
from pathlib import Path

import pytest

_BACK_ROOT = Path(__file__).resolve().parents[2]
if str(_BACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACK_ROOT))
# Avoid api.__init__ starting the FastAPI server and loading the index for this
# contract-only test module.
if "api" not in sys.modules:
    api_package = types.ModuleType("api")
    api_package.__path__ = [str(_BACK_ROOT / "api")]
    sys.modules["api"] = api_package

from api.orchestration import (  # noqa: E402
    MetadataConstraints,
    MetadataConstraint,
    OrchestrationPlan,
    OrchestrationPlanOutputError,
    TemporalConstraint,
    filter_retrieval_candidates,
    compact_history,
    build_prompt,
    plan_once,
    sanitize_plan_for_retrieval,
)


def test_plan_requires_standalone_query_for_retrieval():
    with pytest.raises(ValueError):
        OrchestrationPlan.model_validate({
            "intent": "document_question", "needs_retrieval": True,
            "response_strategy": "answer",
        })


def test_invalid_provider_json_is_rejected_for_safe_fallback():
    with pytest.raises(OrchestrationPlanOutputError):
        plan_once("Question", [], lambda *_args, **_kwargs: "not json", model=None, timeout=1)


def test_temporal_and_sender_constraints_filter_existing_candidates():
    plan = OrchestrationPlan(
        intent="refine_previous_search", needs_retrieval=True,
        retrieval_query="standalone subject", use_history=True, source_types=["email"],
        source_type_provenance={"email": "explicit"},
        temporal_constraints=[TemporalConstraint(mode="exact", date=date(2026, 2, 14), strength="strong", provenance="explicit")],
        metadata_constraints=MetadataConstraints(sender=MetadataConstraint(value="Paul", provenance="explicit")), response_strategy="answer",
    )
    candidates = [
        (0.9, {"source": "email", "document_metadata": {"sender": "Paul Martin", "date": "2026-02-14T09:00:00Z"}}),
        (0.99, {"source": "email", "document_metadata": {"sender": "Paul Martin", "date": "2026-02-13T09:00:00Z"}}),
        (0.95, {"source": "pdf", "document_metadata": {"sender": "Paul Martin", "date": "2026-02-14T09:00:00Z"}}),
    ]
    assert filter_retrieval_candidates(candidates, plan) == [candidates[0]]


def test_recent_is_an_ordering_hint_not_a_fabricated_date_filter():
    plan = OrchestrationPlan(
        intent="refine_previous_search", needs_retrieval=True, retrieval_query="subject",
        source_types=["email"], source_type_provenance={"email": "explicit"}, temporal_constraints=[TemporalConstraint(mode="recent", provenance="inferred")],
        response_strategy="answer",
    )
    old = (0.99, {"source": "email", "document_metadata": {"date": "2026-01-01T00:00:00Z"}})
    new = (0.5, {"source": "email", "document_metadata": {"date": "2026-08-01T00:00:00Z"}})
    assert filter_retrieval_candidates([old, new], plan) == [new, old]


def test_personal_document_question_plan_uses_local_retrieval():
    raw = json.dumps({
        "intent": "document_question", "needs_retrieval": True,
        "retrieval_query": "statut personnel du dossier", "use_history": False,
        "source_types": [], "temporal_constraints": [], "metadata_constraints": {},
        "response_strategy": "answer",
    })
    plan = plan_once("Quel est le statut de mon dossier ?", [], lambda *_args, **_kwargs: raw, model=None, timeout=1)
    assert plan.needs_retrieval is True
    assert plan.intent == "document_question"


def test_general_question_plan_can_skip_retrieval():
    raw = json.dumps({
        "intent": "general_question", "needs_retrieval": False,
        "use_history": False, "source_types": [], "temporal_constraints": [],
        "metadata_constraints": {}, "response_strategy": "general_answer",
    })
    plan = plan_once("Explique un principe général.", [], lambda *_args, **_kwargs: raw, model=None, timeout=1)
    assert plan.needs_retrieval is False


def test_source_refinement_reuses_active_subject_and_email_scope():
    history = [
        {"role": "user", "content": "Quelle décision a été prise concernant ma demande ?"},
        {"role": "assistant", "content": "Je vérifie le contexte."},
    ]
    raw = json.dumps({
        "intent": "refine_previous_search", "needs_retrieval": True,
        "retrieval_query": "décision concernant la demande discutée précédemment",
        "use_history": True, "reuse_previous_subject": True, "source_types": ["email"],
        "temporal_constraints": [], "metadata_constraints": {}, "response_strategy": "answer",
    })
    plan = plan_once("Regarde dans les mails.", history, lambda *_args, **_kwargs: raw, model=None, timeout=1)
    assert plan.retrieval_query != "Regarde dans les mails."
    assert plan.use_history and plan.reuse_previous_subject
    assert plan.source_types == ["email"]


def test_unresolved_followup_asks_for_clarification_instead_of_vague_search():
    raw = json.dumps({
        "intent": "conversation", "needs_retrieval": False,
        "use_history": True, "source_types": [], "temporal_constraints": [],
        "metadata_constraints": {}, "response_strategy": "ask_for_missing_information",
    })
    plan = plan_once("Cherche ça.", [], lambda *_args, **_kwargs: raw, model=None, timeout=1)
    assert plan.needs_retrieval is False
    assert plan.response_strategy == "ask_for_missing_information"


def test_compact_history_keeps_subject_before_short_followup():
    view = compact_history([
        {"role": "user", "content": "Sujet documentaire détaillé."},
        {"role": "assistant", "content": "Réponse intermédiaire."},
        {"role": "user", "content": "Oui, cherche ça."},
    ])
    assert view["active_subject_candidates"] == ["Sujet documentaire détaillé.", "Oui, cherche ça."]


def test_system_prompt_defines_auxilium_role_capabilities_and_priority():
    prompt = build_prompt("Ambiguous internal question", [])
    assert "Auxilium's internal orchestrator" in prompt
    assert "not the final assistant" in prompt
    assert "indexed local documents and indexed emails" in prompt
    assert "Prefer the user's own information" in prompt
    assert "When uncertain between document_question and general_question, prefer document_question" in prompt
    assert "no live mailbox outside synchronized/indexed data" in prompt


def test_system_prompt_includes_smalltalk_documentary_followup_and_general_examples():
    prompt = build_prompt("Question", [])
    assert '"Hello" -> conversation' in prompt
    assert '"Can model X be adapted for a harsh environment?" -> document_question' in prompt
    assert '"Was my request approved?" -> document_question' in prompt
    assert '"Search recent emails instead" with an active subject -> refine_previous_search' in prompt
    assert '"Explain REST APIs" -> general_question' in prompt


def test_ambiguous_documentary_plan_prefers_retrieval():
    raw = json.dumps({
        "intent": "document_question", "needs_retrieval": True,
        "retrieval_query": "internal status relevant to the question", "use_history": False,
        "source_types": [], "temporal_constraints": [], "metadata_constraints": {},
        "response_strategy": "answer",
    })
    plan = plan_once("Ambiguous internal status", [], lambda *_args, **_kwargs: raw, model=None, timeout=1)
    assert plan.intent == "document_question"
    assert plan.needs_retrieval is True


def test_person_mentioned_as_subject_is_not_a_hard_sender_filter():
    raw = OrchestrationPlan(
        intent="document_question", needs_retrieval=True,
        retrieval_query="statut de la demande d'Alice",
        metadata_constraints=MetadataConstraints(sender=MetadataConstraint(value="Alice", provenance="inferred")),
        response_strategy="answer",
    )
    sanitized = sanitize_plan_for_retrieval(raw, user_message="I am Alice. What is the status of my request?")
    assert sanitized.plan.metadata_constraints.sender is None
    assert sanitized.soft_preferences[0]["kind"] == "metadata.sender"
    candidates = [(0.9, {"source": "email", "document_metadata": {"sender": "Service des dossiers"}})]
    assert filter_retrieval_candidates(candidates, sanitized.plan) == candidates


def test_explicit_sender_constraint_is_kept_as_a_hard_filter():
    raw = OrchestrationPlan(
        intent="document_question", needs_retrieval=True,
        retrieval_query="mails d'Alice concernant ma demande",
        metadata_constraints=MetadataConstraints(sender=MetadataConstraint(value="Alice", provenance="explicit")),
        response_strategy="answer",
    )
    sanitized = sanitize_plan_for_retrieval(raw, user_message="Search emails sent by Alice about my request.")
    assert sanitized.plan.metadata_constraints.sender.value == "Alice"
    assert sanitized.hard_filters[0]["kind"] == "metadata.sender"


def test_inferred_constraints_cannot_turn_candidates_into_no_results():
    raw = OrchestrationPlan(
        intent="document_question", needs_retrieval=True, retrieval_query="statut du dossier",
        source_types=["email"],
        temporal_constraints=[TemporalConstraint(mode="exact", date=date(2026, 2, 14), provenance="inferred")],
        metadata_constraints=MetadataConstraints(sender=MetadataConstraint(value="Alice", provenance="inferred")),
        response_strategy="answer",
    )
    candidates = [(0.9, {"source": "pdf", "document_metadata": {"sender": "Service", "date": "2026-02-13T00:00:00Z"}})]
    assert filter_retrieval_candidates(candidates, sanitize_plan_for_retrieval(raw, user_message="What is the status?").plan) == candidates


def test_explicit_pdf_scope_is_kept_as_a_hard_filter():
    raw = OrchestrationPlan(
        intent="refine_previous_search", needs_retrieval=True, retrieval_query="sujet actif",
        source_types=["pdf"], source_type_provenance={"pdf": "explicit"}, response_strategy="answer",
    )
    assert sanitize_plan_for_retrieval(raw, user_message="Look only in PDFs.").plan.source_types == ["pdf"]


def test_invalid_plan_keeps_raw_output_and_field_errors_for_debugging():
    raw = '{"intent":"document_question","needs_retrieval":true,"retrieval_query":"ok","response_strategy":"answer","source_types":["unknown"]}'
    with pytest.raises(OrchestrationPlanOutputError) as exc_info:
        plan_once("Question", [], lambda *_args, **_kwargs: raw, model=None, timeout=1)
    assert exc_info.value.raw_model_output == raw
    assert exc_info.value.validation_error_details


def test_null_optional_structures_are_normalized_but_invalid_values_still_fail():
    raw = json.dumps({
        "intent": "refine_previous_search", "needs_retrieval": True,
        "retrieval_query": "status of the active request", "use_history": True,
        "source_types": None, "source_type_provenance": None,
        "temporal_constraints": None, "metadata_constraints": None,
        "response_strategy": "answer",
    })
    plan = plan_once("I think I received an answer", [], lambda *_args, **_kwargs: raw, model=None, timeout=1)
    assert plan.source_types == []
    assert plan.source_type_provenance == {}
    assert plan.temporal_constraints == []
    assert plan.metadata_constraints == MetadataConstraints()
    assert plan._raw_model_output == raw

    invalid = raw.replace('"source_types": null', '"source_types": "email"')
    with pytest.raises(OrchestrationPlanOutputError):
        plan_once("Question", [], lambda *_args, **_kwargs: invalid, model=None, timeout=1)


def _metadata_plan(field: str, provenance: str = "explicit") -> OrchestrationPlan:
    return OrchestrationPlan(
        intent="document_question", needs_retrieval=True, retrieval_query="status of Alice's request",
        metadata_constraints=MetadataConstraints(**{field: MetadataConstraint(value="Alice", provenance=provenance)}),
        response_strategy="answer",
    )


def test_explicit_llm_sender_is_downgraded_without_an_sending_relation():
    sanitized = sanitize_plan_for_retrieval(_metadata_plan("sender"), user_message="I am Alice. What is the status of my request?")
    assert sanitized.plan.metadata_constraints.sender is None
    assert sanitized.hard_filters == []
    decision = sanitized.soft_preferences[0]
    assert decision["llm_provenance"] == "explicit"
    assert decision["validated_provenance"] == "inferred"
    assert decision["validation_reason"] == "no explicit sender restriction found in user request"


def test_sender_and_recipient_hard_filters_need_their_own_relations():
    sender = sanitize_plan_for_retrieval(_metadata_plan("sender"), user_message="Regarde les mails envoyés par Alice.")
    recipient = sanitize_plan_for_retrieval(_metadata_plan("recipient"), user_message="Regarde les mails reçus par Alice.")
    assert sender.plan.metadata_constraints.sender.value == "Alice"
    assert recipient.plan.metadata_constraints.recipient.value == "Alice"


def test_entity_in_subject_never_becomes_sender_or_recipient_filter():
    for field in ("sender", "recipient"):
        sanitized = sanitize_plan_for_retrieval(_metadata_plan(field), user_message="Look at exchanges concerning Alice.")
        assert getattr(sanitized.plan.metadata_constraints, field) is None
        assert not sanitized.hard_filters


def test_explicit_date_restriction_can_be_hard_but_current_status_is_not():
    dated = OrchestrationPlan(
        intent="document_question", needs_retrieval=True, retrieval_query="project file since January",
        temporal_constraints=[TemporalConstraint(mode="after", date=date(2026, 1, 1), provenance="explicit")], response_strategy="answer",
    )
    explicit = sanitize_plan_for_retrieval(dated, user_message="Regarde le dossier depuis le 1er janvier.")
    assert len(explicit.plan.temporal_constraints) == 1
    assert explicit.hard_filters[0]["validated_provenance"] == "explicit"

    recent = OrchestrationPlan(
        intent="document_question", needs_retrieval=True, retrieval_query="current project status",
        temporal_constraints=[TemporalConstraint(mode="recent", provenance="explicit")], response_strategy="answer",
    )
    soft = sanitize_plan_for_retrieval(recent, user_message="What is the current status of the project?")
    assert not soft.hard_filters
    assert soft.soft_preferences[0]["validated_provenance"] == "inferred"
