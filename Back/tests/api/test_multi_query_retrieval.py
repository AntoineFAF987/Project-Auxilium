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

from api.multi_query_retrieval import (  # noqa: E402
    ResolvedUserTask, _canonical, _specific_anchors, build_cross_language_query, build_retrieval_queries, canonicalize_information_need, detect_query_language,
    evaluate_cross_language_query, reciprocal_rank_fusion, resolve_retrieval_query,
    validate_cross_language_query,
)


def _chunk(uid, document):
    return {"chunk_uid": uid, "document_id": document, "chunk_id": 0, "text": document}


def test_original_query_is_always_retained_alongside_orchestrator_rewrite():
    queries = build_retrieval_queries(
        original_query="Do I finally have permission to leave the country?",
        orchestrator_query="determine the current status of a travel request",
    )
    assert queries[0][0] == "original_autonomous"
    assert any(kind == "orchestrator" for kind, _ in queries)
    assert len(queries) <= 3


def test_original_query_precedes_a_distinct_orchestrator_rewrite_for_technical_anchors():
    queries = build_retrieval_queries(
        original_query="2420 2334",
        orchestrator_query="Type 2334 actuator selection",
    )
    assert queries[:2] == [
        ("original_autonomous", "2420 2334"),
        ("orchestrator", "Type 2334 actuator selection"),
    ]


def test_normalized_variant_does_not_duplicate_original_or_rewrite():
    queries = build_retrieval_queries(
        original_query="Type 2334 actuator selection",
        orchestrator_query="Type 2334 actuator selection",
    )
    assert queries == [("original_autonomous", "Type 2334 actuator selection")]


def test_obvious_word_order_variant_does_not_duplicate_rewrite():
    queries = build_retrieval_queries(
        original_query="Type 2334 actuator selection",
        orchestrator_query="selection actuator Type 2334",
    )
    assert queries == [("original_autonomous", "Type 2334 actuator selection")]


def test_rrf_preserves_candidate_found_only_by_original_wording():
    doc_a = _chunk("a", "request authorization")
    doc_b = _chunk("b", "restriction lifted")
    fused = reciprocal_rank_fusion([
        ("original", [(0.9, doc_b), (0.8, doc_a)]),
        ("orchestrator", [(0.95, doc_a)]),
    ])
    by_uid = {meta["chunk_uid"]: meta for _score, meta in fused}
    assert "b" in by_uid
    assert by_uid["b"]["retrieved_by"] == ["original"]
    assert by_uid["b"]["per_query_rank"] == {"original": 1}
    assert by_uid["a"]["retrieved_by"] == ["original", "orchestrator"]


def test_duplicate_rewrites_do_not_trigger_duplicate_searches():
    queries = build_retrieval_queries(original_query="Model 3725 maximum temperature", orchestrator_query="Model 3725 maximum temperature")
    assert [kind for kind, _ in queries] == ["original_autonomous"]


def test_canonical_information_need_converges_productive_french_isation_forms():
    variants = [
        "Peut-on tropicaliser le modèle X ?",
        "Le modèle X est-il tropicalisable ?",
        "Le modèle X peut-il être tropicalisé ?",
        "Quelle est la possibilité de tropicalisation du modèle X ?",
    ]
    canonical = [canonicalize_information_need(value) for value in variants]
    assert all(value and "tropicalisation" in value for value in canonical)


def test_canonical_information_need_preserves_exact_technical_entities_byte_for_byte():
    query = "Le 3731 peut-il être tropicalisé avec 3730-3 HV02 82.7 DN50 PS-AMS KG2 ?"
    result = canonicalize_information_need(query, exact_entities=tuple(_specific_anchors(query)))
    assert result
    for entity in ("3731", "3730-3", "HV02", "82.7", "DN50", "PS-AMS", "KG2"):
        assert entity in result
    assert "3730 " not in result


def test_canonical_information_need_preserves_multiword_and_structured_terms():
    dead_band = "Quelle est la dead band du PS AMS ?"
    price = "Quel est le prix de vente du KG2 DN50 ?"
    dead_band_result = canonicalize_information_need(dead_band, exact_entities=tuple(_specific_anchors(dead_band)))
    price_result = canonicalize_information_need(price, exact_entities=tuple(_specific_anchors(price)))
    assert dead_band_result and "dead band" in dead_band_result and "PS AMS" in dead_band_result
    assert price_result and "prix de vente" in price_result
    assert "KG2" in price_result and "DN50" in price_result


def test_compact_documentary_query_does_not_receive_a_weaker_canonical_variant():
    assert canonicalize_information_need("PS AMS dead band", exact_entities=("PS AMS",)) == "PS AMS dead band"


def test_canonical_information_need_keeps_current_state_semantics_outside_normalization():
    result = canonicalize_information_need("Ma demande a-t-elle été validée ?")
    assert result and "valid" in result.casefold()
    task = ResolvedUserTask(
        factual_query="Ma demande a-t-elle été validée ?", exact_entities=(), information_need="validation",
        query_semantics="current_state", canonical_information_need_query=result,
    )
    assert task.query_semantics == "current_state"


def _canonical_core(query: str) -> set[str]:
    return set(_canonical(query).split()) - {"de", "du", "positionneur", "modele"}


def test_canonical_information_need_removes_french_interrogative_inversion_generically():
    variants = [
        "Le positionneur 3731 peut-il être tropicalisé ?",
        "Le positionneur 3731 est-il tropicalisable ?",
        "Peut-on tropicaliser le positionneur 3731 ?",
        "Est-ce possible de tropicaliser le positionneur 3731 ?",
        "Quelle est la possibilité de tropicalisation du positionneur 3731 ?",
    ]
    canonical = [
        canonicalize_information_need(value, exact_entities=tuple(_specific_anchors(value)))
        for value in variants
    ]
    assert all(value and "3731" in value and "tropicalisation" in value for value in canonical)
    assert all("peut-il" not in value.casefold() and "est-il" not in value.casefold() for value in canonical)
    assert all("tropicalisation" in _canonical_core(value) for value in canonical)


def test_syntactic_canonicalization_preserves_modal_and_technical_meaning():
    mandatory = "Est-il obligatoire de remplacer le produit X ?"
    compatible = "Le modèle X est-il compatible avec Y ?"
    price = "Le prix de vente du KG2 DN50 est-il disponible ?"
    mandatory_result = canonicalize_information_need(mandatory)
    compatible_result = canonicalize_information_need(compatible)
    price_result = canonicalize_information_need(price, exact_entities=tuple(_specific_anchors(price)))
    assert mandatory_result and all(term in mandatory_result.casefold() for term in ("obligatoire", "remplacer", "x"))
    assert compatible_result and all(term in compatible_result.casefold() for term in ("x", "compatible", "y"))
    assert price_result and "prix de vente" in price_result and "KG2" in price_result and "DN50" in price_result
    assert "disponible" in price_result.casefold()


def test_syntactic_canonicalization_keeps_ps_ams_documentary_terms():
    query = "Quelle est la dead band du PS AMS ?"
    result = canonicalize_information_need(query, exact_entities=tuple(_specific_anchors(query)))
    assert result and "dead band" in result and "PS AMS" in result


def test_followup_uses_resolved_subject_not_literal_conversational_wording():
    resolved = resolve_retrieval_query(
        raw_user_message="Do you find anything in my local sources?",
        orchestrator_query="Project Orion",
        history=[{"role": "user", "content": "What is Project Orion?"}],
    )
    queries = build_retrieval_queries(
        original_query="Do you find anything in my local sources?", orchestrator_query="Project Orion",
        resolved_query=resolved, follow_up=True,
    )
    assert resolved == "Project Orion"
    assert queries == [("resolved_followup", "Project Orion")]


def test_followup_preserves_explicit_new_name_filter():
    resolved = resolve_retrieval_query(
        raw_user_message="And in Pierre's emails?", orchestrator_query="Project Orion", history=[],
    )
    assert "Project Orion" in resolved
    assert "Pierre" in resolved


def test_followup_keeps_user_wording_and_resolved_hv02_subject():
    resolved = resolve_retrieval_query(
        raw_user_message="Et pour le HV02 ?",
        orchestrator_query="Certification NACE de la version HV",
        history=[],
    )
    queries = build_retrieval_queries(
        original_query="Et pour le HV02 ?",
        orchestrator_query="Certification NACE de la version HV",
        resolved_query=resolved,
        follow_up=True,
    )
    assert queries[0][0] == "resolved_followup"
    assert "HV02" in queries[0][1]


def test_conversational_followup_does_not_become_a_documentary_query():
    resolved = resolve_retrieval_query(
        raw_user_message="cherche encore", orchestrator_query="Courriel du 28 août concernant Antoine",
    )
    queries = build_retrieval_queries(
        original_query="cherche encore", orchestrator_query="Courriel du 28 août concernant Antoine",
        resolved_query=resolved, follow_up=True,
    )
    assert all(query != "cherche encore" for _, query in queries)
    assert queries[0] == ("resolved_followup", "Courriel du 28 août concernant Antoine")


def test_followup_keeps_explicit_email_date_without_conversational_vocabulary():
    raw = "tu as un mail du 28 qui donne la réponse"
    resolved = resolve_retrieval_query(
        raw_user_message=raw, orchestrator_query="Antoine autorisé à quitter le territoire",
    )
    queries = build_retrieval_queries(
        original_query=raw, orchestrator_query="Antoine autorisé à quitter le territoire",
        resolved_query=resolved, follow_up=True,
    )
    assert "28" in queries[0][1]
    assert "mail" in queries[0][1].casefold()
    assert all(raw != query for _, query in queries)


def test_followup_ignores_incidental_conversational_words_present_in_corpus():
    raw = "mais cherche son contenu fossile draeger"
    resolved = resolve_retrieval_query(raw_user_message=raw, orchestrator_query="Courriel du 28 août concernant Antoine")
    queries = build_retrieval_queries(
        original_query=raw, orchestrator_query="Courriel du 28 août concernant Antoine",
        resolved_query=resolved, follow_up=True,
    )
    rendered = " ".join(query.casefold() for _, query in queries)
    assert "fossile" not in rendered
    assert "draeger" not in rendered


def test_french_ps_ams_gets_a_concise_english_documentary_variant():
    original = "Quelle est la bande morte des actionneurs PS AMS ?"
    variant = build_cross_language_query(
        original_user_query=original,
        orchestrator_query="bande morte actionneur PS AMS",
        detected_language=detect_query_language(original),
        query_semantics="fact_lookup",
    )
    queries = build_retrieval_queries(
        original_query=original,
        orchestrator_query="bande morte actionneur PS AMS",
        cross_language_query=variant,
    )
    assert detect_query_language(original) == "fr"
    assert variant == "PS AMS actuator dead band"
    assert ("cross_language", "PS AMS actuator dead band") in queries


def test_english_ps_ams_does_not_get_a_redundant_english_variant():
    question = "What is the dead band of a PS AMS actuator?"
    decision = evaluate_cross_language_query(
        original_user_query=question, orchestrator_query=question,
        detected_language=detect_query_language(question), query_semantics="fact_lookup",
    )
    assert detect_query_language(question) == "en"
    assert decision.query is None
    assert decision.rejection_reason == "source_language_not_french"


def test_cross_language_variant_preserves_numeric_and_model_anchors():
    variant = build_cross_language_query(
        original_user_query="Quelle est la température max du 82.7 HV02 ?",
        orchestrator_query="température maximale 82.7 HV02",
        detected_language="fr", query_semantics="fact_lookup",
    )
    assert variant is not None
    assert "82.7" in variant
    assert "HV02" in variant


def test_cross_language_variant_preserves_product_code():
    variant = build_cross_language_query(
        original_user_query="Quelle est la limite du 2420 ?",
        orchestrator_query="limite 2420",
        detected_language="fr", query_semantics="fact_lookup",
    )
    assert variant is not None
    assert "2420" in variant


def test_cross_language_variant_is_not_added_when_duplicate():
    queries = build_retrieval_queries(
        original_query="PS AMS actuator dead band",
        orchestrator_query="PS AMS actuator dead band",
        cross_language_query="dead band actuator PS AMS",
    )
    assert [kind for kind, _query in queries] == ["original_autonomous"]


def test_non_documentary_query_has_no_cross_language_expansion():
    decision = evaluate_cross_language_query(
        original_user_query="Bonjour, comment vas-tu ?", orchestrator_query=None,
        detected_language="fr", query_semantics=None,
    )
    assert decision.query is None
    assert decision.rejection_reason == "non_documentary_query"


def test_followup_cross_language_uses_resolved_subject_not_raw_instruction():
    raw_followup = "cherche encore"
    variant = build_cross_language_query(
        original_user_query=raw_followup,
        orchestrator_query="bande morte actionneur PS AMS",
        detected_language=detect_query_language(raw_followup), query_semantics="fact_lookup",
    )
    assert detect_query_language(raw_followup) == "fr"
    assert variant == "PS AMS actuator dead band"


def test_semantic_orchestrator_variant_preserves_zone_morte_information_need():
    question = "Quelle est la zone morte des actionneurs PS AMS ?"
    decision = evaluate_cross_language_query(
        original_user_query=question,
        orchestrator_query="zone morte actionneur PS AMS",
        detected_language="fr", query_semantics="fact_lookup",
        proposed_query="PS AMS actuator dead band",
    )
    assert decision.query == "PS AMS actuator dead band"
    assert decision.information_need_retained is True
    assert decision.validation_passed is True


def test_procedure_variant_retains_adjustment_intent():
    question = "Comment régler la bande morte des actionneurs PS AMS ?"
    decision = evaluate_cross_language_query(
        original_user_query=question,
        orchestrator_query="réglage bande morte actionneur PS AMS",
        detected_language="fr", query_semantics="procedure",
    )
    assert decision.query == "PS AMS actuator dead band adjustment"
    assert decision.semantic_intent_retained is True


def test_fact_lookup_and_procedure_do_not_collapse_to_same_information_need():
    fact = build_cross_language_query(
        original_user_query="Quelle est la bande morte des actionneurs PS AMS ?",
        orchestrator_query="bande morte actionneur PS AMS",
        detected_language="fr", query_semantics="fact_lookup",
    )
    procedure = build_cross_language_query(
        original_user_query="Comment régler la bande morte des actionneurs PS AMS ?",
        orchestrator_query="réglage bande morte actionneur PS AMS",
        detected_language="fr", query_semantics="procedure",
    )
    assert fact == "PS AMS actuator dead band"
    assert procedure == "PS AMS actuator dead band adjustment"


def test_overly_generic_candidate_is_rejected_when_information_need_is_lost():
    decision = validate_cross_language_query(
        candidate="HV02 valve", anchors={"HV02"}, query_semantics="fact_lookup",
    )
    assert decision.query is None
    assert decision.rejection_reason == "information_need_lost"


def test_proposed_temperature_query_preserves_all_anchors_and_property():
    decision = evaluate_cross_language_query(
        original_user_query="Quelle est la température max du 82.7 HV02 ?",
        orchestrator_query="température maximale 82.7 HV02",
        detected_language="fr", query_semantics="fact_lookup",
        proposed_query="82.7 HV02 maximum operating temperature",
    )
    assert decision.query is not None
    assert "82.7" in decision.query and "HV02" in decision.query


def test_followup_procedure_uses_resolved_subject_and_not_raw_pronoun():
    decision = evaluate_cross_language_query(
        original_user_query="Et comment la régler ?",
        orchestrator_query="PS AMS dead band",
        detected_language="fr", query_semantics="procedure",
        proposed_query="PS AMS actuator dead band adjustment",
    )
    assert decision.query == "PS AMS actuator dead band adjustment"


def test_semantic_procedure_and_decision_queries_are_validated_without_product_rules():
    configuration = evaluate_cross_language_query(
        original_user_query="Comment configurer le 3730 ?",
        orchestrator_query="configuration 3730",
        detected_language="fr", query_semantics="procedure",
        proposed_query="3730 configuration procedure",
    )
    decision = evaluate_cross_language_query(
        original_user_query="Ma demande a-t-elle été acceptée ?",
        orchestrator_query="statut de décision de la demande",
        detected_language="fr", query_semantics="decision",
        proposed_query="final decision approval status",
    )
    assert configuration.query == "3730 configuration procedure"
    assert decision.query == "final decision approval status"
