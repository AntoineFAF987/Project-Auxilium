from eval.generation_frozen_context import FIXTURES, SYNTHETIC_FIXTURES, VARIANTS, evaluate_answer, payload_for_variant


def test_harness_captures_current_payload_without_provider_call():
    payload = payload_for_variant(FIXTURES[0], VARIANTS[0])
    assert payload["messages"][0]["role"] == "system"
    assert "CONTEXTE:" in payload["messages"][-1]["content"]
    assert FIXTURES[0].context in payload["messages"][-1]["content"]


def test_prompt_variant_changes_only_system_instructions():
    baseline = payload_for_variant(FIXTURES[0], VARIANTS[0])
    variant = payload_for_variant(FIXTURES[0], VARIANTS[1])
    assert baseline["messages"][-1] == variant["messages"][-1]
    assert len(variant["messages"][0]["content"]) > len(baseline["messages"][0]["content"])


def test_fixture_evaluator_is_pattern_based_and_detects_forbidden_claims():
    fixture = FIXTURES[0]
    answer = (
        "La tropicalisation supplémentaire n'est plus possible. La carte est déjà vernie, sans être une tropicalisation complète. "
        "Le changement de technologie du détecteur l'empêche et un vernis supplémentaire modifierait le comportement."
    )
    result = evaluate_answer(fixture, answer)
    assert result["required_fact_recall"] == 1.0
    assert result["forbidden_claim_count"] == 0


def test_fixture_evaluator_keeps_related_entity_safety_without_false_positive():
    fixture = SYNTHETIC_FIXTURES[1]
    answer = "Aucune information directe ne concerne X371. La limitation du X370 ne peut pas être attribuée au X371."
    result = evaluate_answer(fixture, answer)
    assert result["required_fact_recall"] == 1.0
    assert result["forbidden_claim_count"] == 0


def test_synthetic_fixtures_preserve_evaluation_contract_without_real_identifiers():
    assert len(SYNTHETIC_FIXTURES) == len(FIXTURES) == 5
    for real, synthetic in zip(FIXTURES, SYNTHETIC_FIXTURES):
        assert len(real.required_patterns) == len(synthetic.required_patterns)
        assert len(real.forbidden_patterns) == len(synthetic.forbidden_patterns)
        assert "3730" not in synthetic.context
        assert "3731" not in synthetic.context
        assert "PS AMS" not in synthetic.context
        assert "JAUMOUILL" not in synthetic.context.upper()
        assert "KG2" not in synthetic.context
