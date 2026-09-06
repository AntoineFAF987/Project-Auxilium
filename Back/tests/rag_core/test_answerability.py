from rag_core.context_sufficiency import evaluate_answerability


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
