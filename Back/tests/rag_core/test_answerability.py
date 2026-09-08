from rag_core.context_sufficiency import evaluate_answerability, evaluate_exact_entity_support


def _block(text, document="doc-1"):
    return {"text": text, "document_id": document}


def test_direct_precise_anchor_is_answerable():
    decision = evaluate_answerability(
        query="La version HV02 est-elle certifiee NACE ?",
        evidence_mode="direct", context_is_relevant=True, evidence_sufficient=True,
        reranker_accepted=True,
        blocks=[_block("La version HV02 est certifiee NACE.")],
    )
    assert decision.status == "answerable"


def test_neighbouring_version_is_never_answerable_for_precise_anchor():
    decision = evaluate_answerability(
        query="La version HV02 est-elle certifiee NACE ?",
        evidence_mode="related", context_is_relevant=True, evidence_sufficient=True,
        reranker_accepted=True,
        blocks=[_block("La version HV01 est certifiee NACE.")],
    )
    assert decision.status in {"partial", "unanswerable"}
    assert decision.status != "answerable"
    assert "hv02" in decision.requested_anchors


def test_two_fact_question_with_only_one_fact_is_partial():
    decision = evaluate_answerability(
        query="Quel est le materiau et la pression maximale ?",
        evidence_mode="direct", context_is_relevant=True, evidence_sufficient=False,
        reranker_accepted=True,
        blocks=[_block("Le corps est en acier inoxydable.")],
    )
    assert decision.status == "partial"


def test_no_context_is_unanswerable():
    decision = evaluate_answerability(
        query="Quelle est la pression maximale ?",
        evidence_mode="none", context_is_relevant=False, evidence_sufficient=False,
        reranker_accepted=False, blocks=[],
    )
    assert decision.status == "unanswerable"


def test_lexical_noise_without_information_need_alignment_is_unanswerable():
    decision = evaluate_answerability(
        query="Quel est le materiau du Type 2420 ?",
        evidence_mode="none", context_is_relevant=False, evidence_sufficient=False,
        reranker_accepted=True,
        blocks=[_block("Le Type 2422 apparait dans un calendrier de livraison.")],
    )
    assert decision.status == "unanswerable"


def test_neighbouring_type_is_never_answerable_for_precise_request():
    decision = evaluate_answerability(
        query="Quel est le materiau du Type 2420 ?",
        evidence_mode="related", context_is_relevant=True, evidence_sufficient=True,
        reranker_accepted=True,
        blocks=[_block("Le Type 2422 est disponible avec plusieurs materiaux.")],
    )
    assert decision.status != "answerable"
    assert decision.aspect_coverage_override_applied is False


def test_complete_semantic_aspect_coverage_overrides_low_lexical_coverage_only():
    decision = evaluate_answerability(
        query="Antoine is he allowed to leave the territory?",
        evidence_mode="direct", context_is_relevant=True, evidence_sufficient=True,
        reranker_accepted=True,
        blocks=[_block("The request was granted and the travel restriction was lifted.")],
        requested_aspects=["final decision / approval status"],
        supported_aspects=["final decision / approval status"],
        missing_aspects=[],
    )
    assert decision.query_term_coverage < 0.55
    assert decision.status == "answerable"
    assert decision.semantic_aspect_coverage_complete is True
    assert decision.aspect_coverage_override_applied is True
    assert decision.override_reason == "all_requested_aspects_supported"


def test_english_evidence_uses_best_equivalent_query_coverage_not_french_ratio():
    decision = evaluate_answerability(
        query="Quelle est la bande morte des actionneurs PS AMS ?",
        query_variants=["PS AMS actuator dead band"],
        evidence_mode="direct", context_is_relevant=True, evidence_sufficient=True,
        reranker_accepted=True,
        blocks=[_block("The dead band of PS-AMS actuators is adjustable between 0.5% and 5%.")],
    )
    assert decision.query_term_coverage >= 0.75
    assert decision.status == "answerable"


def test_native_english_query_keeps_existing_coverage_behavior():
    decision = evaluate_answerability(
        query="What is the actuator dead band?",
        evidence_mode="direct", context_is_relevant=True, evidence_sufficient=True,
        reranker_accepted=True,
        blocks=[_block("The actuator dead band is adjustable between 0.5% and 5%.")],
    )
    assert decision.query_term_coverage >= 0.75


def test_missing_aspect_cannot_use_semantic_override():
    decision = evaluate_answerability(
        query="HV02 NACE certification and temperature limits",
        evidence_mode="direct", context_is_relevant=True, evidence_sufficient=True,
        reranker_accepted=True, blocks=[_block("HV02 has NACE certification.")],
        requested_aspects=["NACE certification", "operating temperature limits"],
        supported_aspects=["NACE certification"],
        missing_aspects=["operating temperature limits"],
    )
    assert decision.status == "partial"
    assert decision.aspect_coverage_override_applied is False


def test_missing_anchor_blocks_semantic_override():
    decision = evaluate_answerability(
        query="HV02 temperature limits",
        evidence_mode="direct", context_is_relevant=True, evidence_sufficient=True,
        reranker_accepted=True, blocks=[_block("HV01 temperature limits are documented.")],
        requested_aspects=["operating temperature limits"],
        supported_aspects=["operating temperature limits"], missing_aspects=[],
    )
    assert decision.status != "answerable"
    assert decision.aspect_coverage_override_applied is False


def test_structural_gap_blocks_semantic_override():
    decision = evaluate_answerability(
        query="Is the request approved?", evidence_mode="direct", context_is_relevant=True,
        evidence_sufficient=False, reranker_accepted=True,
        blocks=[_block("The request was approved.")],
        requested_aspects=["final decision / approval status"],
        supported_aspects=["final decision / approval status"], missing_aspects=[],
    )
    assert decision.status != "answerable"
    assert decision.aspect_coverage_override_applied is False


def test_unresolved_current_state_blocks_semantic_override():
    decision = evaluate_answerability(
        query="What is the current status?", evidence_mode="direct", context_is_relevant=True,
        evidence_sufficient=False, reranker_accepted=True,
        blocks=[_block("The previous status was pending.")],
        requested_aspects=["latest/current state"],
        supported_aspects=["latest/current state"], missing_aspects=[],
    )
    assert decision.status != "answerable"
    assert decision.aspect_coverage_override_applied is False


def _exact_support(question, evidence):
    decision = evaluate_answerability(
        query=question, evidence_mode="related", context_is_relevant=True,
        evidence_sufficient=True, reranker_accepted=True, blocks=[_block(evidence)],
    )
    return evaluate_exact_entity_support(
        requested_anchors=decision.requested_anchors,
        supported_anchors=decision.supported_anchors,
        blocks=[_block(evidence)],
    )


def test_exact_entity_guard_rejects_neighbouring_numeric_reference():
    support = _exact_support("3731 tropicalisation", "3730 tropicalisation is no longer possible")
    assert support.support == "missing"
    assert support.missing_exact_entities == ("3731",)
    assert support.related_only_entities == ("3731",)


def test_exact_entity_guard_accepts_exact_reference():
    support = _exact_support("3731 tropicalisation", "3731 tropicalisation is possible")
    assert support.support == "complete"
    assert support.guard_applied is False


def test_exact_entity_guard_rejects_version_and_size_mismatches():
    version = _exact_support("HV02 maximum temperature", "HV01 maximum temperature 80C")
    size = _exact_support("DN50", "DN80 maximum temperature 80C")
    assert version.support == "missing"
    assert "hv02" in version.missing_exact_entities
    assert size.support == "missing"
    assert "dn50" in size.missing_exact_entities


def test_exact_entity_guard_does_not_transfer_product_support_from_same_size():
    support = _exact_support("KG2 DN50", "KG9 DN50 is available")
    assert support.support == "partial"
    assert "kg2" in support.missing_exact_entities
    assert "dn50" in support.supported_exact_entities


def test_exact_entity_guard_is_partial_for_multiple_requested_entities():
    support = _exact_support("difference HV01 HV02", "HV01 supports feature A")
    assert support.support == "partial"
    assert support.supported_exact_entities == ("hv01",)
    assert support.missing_exact_entities == ("hv02",)


def test_exact_entity_guard_is_not_applicable_to_generic_question():
    support = _exact_support("Comment fonctionne un positionneur ?", "Un positionneur compare une consigne.")
    assert support.support == "not_applicable"
