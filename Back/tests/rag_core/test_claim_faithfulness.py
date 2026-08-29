from rag_core.faithfulness import (
    CitationStatus,
    ClaimStatus,
    extract_answer_claims,
    verify_answer_claims,
)


def _block(text, uid="chunk-1", *, document="doc-1", page=8, section="Tropicalization"):
    return {
        "text": text,
        "chunk_uid": uid,
        "document_id": document,
        "path": f"/docs/{document}.pdf",
        "page": page,
        "section": section,
        "heading_path": ["Products", section],
        "block_ids": [f"{uid}:block"],
    }


def _statuses(review):
    return [item.status for item in review.claims]


def test_claim_extraction_ignores_style_and_recommendations_and_has_stable_ids():
    answer = (
        "Bonjour. Le circuit imprimé du 3730 est déjà verni. "
        "Je recommande de contacter le fabricant. Merci."
    )
    first = extract_answer_claims(answer)
    second = extract_answer_claims(answer)
    assert [claim.text for claim in first] == ["Le circuit imprimé du 3730 est déjà verni."]
    assert [claim.claim_id for claim in first] == [claim.claim_id for claim in second]


def test_claim_extraction_splits_explicit_cause_from_compound_sentence():
    claims = extract_answer_claims(
        "Le circuit imprimé du 3730 est déjà verni. "
        "La tropicalisation complète n'est plus possible à cause de la nouvelle technologie du détecteur de position."
    )
    assert [claim.text for claim in claims] == [
        "Le circuit imprimé du 3730 est déjà verni.",
        "La tropicalisation complète n'est plus possible.",
        "La cause est la nouvelle technologie du détecteur de position.",
    ]


def test_3730_claims_are_supported_without_caveat_and_map_precise_spans():
    answer = (
        "Le circuit imprimé du 3730 est déjà verni. "
        "La tropicalisation complète du 3730 est impossible. "
        "La cause est la nouvelle technologie du détecteur de position."
    )
    evidence = _block(
        "Le circuit imprimé du 3730 est déjà verni. "
        "La tropicalisation complète du 3730 est impossible. "
        "La cause est la nouvelle technologie du détecteur de position.",
        uid="doc-3730:c7",
    )
    review = verify_answer_claims(answer, [evidence], use_nli=False)
    assert _statuses(review) == [ClaimStatus.SUPPORTED] * 3
    assert review.status == "OK"
    assert all(item.evidence[0].chunk_uid == "doc-3730:c7" for item in review.claims)
    assert all(item.evidence[0].page == 8 for item in review.claims)
    assert all(item.evidence[0].start < item.evidence[0].end for item in review.claims)


def test_3725_generalization_is_not_supported_by_3730_evidence():
    review = verify_answer_claims(
        "La tropicalisation complète du 3725 est impossible.",
        [_block("La tropicalisation complète du 3730 est impossible.")],
        use_nli=False,
    )
    assert _statuses(review) == [ClaimStatus.UNSUPPORTED]
    assert review.caveat_required is True


def test_contradiction_detects_opposite_polarity():
    review = verify_answer_claims(
        "La tropicalisation complète du 3730 est possible.",
        [_block("La tropicalisation complète du 3730 est impossible.")],
        use_nli=False,
    )
    assert _statuses(review) == [ClaimStatus.CONTRADICTED]


def test_contradiction_detects_curly_apostrophe_negation():
    review = verify_answer_claims(
        "La tropicalisation complète du 3730 est possible.",
        [_block("La tropicalisation complète du 3730 n’est plus possible.")],
        use_nli=False,
    )
    assert _statuses(review) == [ClaimStatus.CONTRADICTED]


def test_multiple_claims_map_to_distinct_sources():
    review = verify_answer_claims(
        "PostgreSQL 16 réduit le CPU de 11 %. Redis 7.2 est approuvé pour la production.",
        [
            _block("PostgreSQL 16 réduit le CPU de 11 %.", "pg:c1", document="postgres"),
            _block("Redis 7.2 est approuvé pour la production.", "redis:c2", document="redis"),
        ],
        use_nli=False,
    )
    assert _statuses(review) == [ClaimStatus.SUPPORTED, ClaimStatus.SUPPORTED]
    assert [item.evidence[0].document_id for item in review.claims] == ["postgres", "redis"]


def test_supported_claim_with_wrong_citation_is_flagged():
    review = verify_answer_claims(
        "Le 3730 possède un circuit imprimé verni.",
        [
            _block("Le 3730 possède un circuit imprimé verni.", "right:c1", document="right"),
            _block("Le 3725 utilise un boîtier en aluminium.", "wrong:c2", document="wrong"),
        ],
        cited_source_indices=[2],
        use_nli=False,
    )
    assert _statuses(review) == [ClaimStatus.SUPPORTED]
    assert review.claims[0].citation_correct is False
    assert review.claims[0].citation_status == CitationStatus.MISMATCHED
    assert review.caveat_required is True


def test_supported_claim_in_another_chunk_of_cited_document_keeps_fact_and_citation_separate():
    review = verify_answer_claims(
        "Le 3730 possède un circuit imprimé verni.",
        [
            _block("Objet : tropicalisation du 3730.", "mail:header", document="mail-3730"),
            _block("Le 3730 possède un circuit imprimé verni.", "mail:body", document="mail-3730"),
        ],
        cited_source_indices=[1],
        use_nli=False,
    )
    assert _statuses(review) == [ClaimStatus.SUPPORTED]
    assert review.claims[0].citation_status == CitationStatus.MATCHED
    assert "another chunk" in (review.claims[0].citation_reason or "")
    assert review.caveat_required is False


def test_real_3725_3730_prudent_answer_has_no_false_partial_evidence():
    answer = (
        "D’après le **contexte fourni**, la tropicalisation du **Du Trovis 3730** n’est plus possible en raison "
        "d’un **changement de technologie au niveau du détecteur de position**. Cette information est explicitement "
        "mentionnée dans le document [1], qui précise que l’application d’un vernis supplémentaire modifierait le "
        "comportement du détecteur.\n\nCependant, **aucune information n’est donnée dans le contexte concernant le "
        "positionneur 3725**. Le document [1] traite uniquement du 3730, et les autres sources ne mentionnent ni ce "
        "modèle ni sa tropicalisation."
    )
    context = _block(
        "Du Trovis 3730 possède déjà un vernis à voir photo Ce n’est pas une tropicalisation complète. "
        "Elle permet de tenir les spec de la fiche T à savoir : –20 à 80 °C pour toutes les exécutions. "
        "La tropicalisation n’est plus possible, le changement de technologie au niveau du détecteur de position "
        "ne le permet plus. L’application d’un vernis supplémentaire change son comportement.",
        "mail-3730:body", document="mail-3730",
    )
    review = verify_answer_claims(answer, [context], cited_source_indices=[1], use_nli=False)
    assert _statuses(review) == [ClaimStatus.SUPPORTED] * 4
    assert review.claims[1].citation_status == CitationStatus.MATCHED
    assert review.claims[2].claim.claim_type == "EVIDENCE_ABSENCE"
    assert review.status == "OK"
    assert review.caveat_required is False


def test_markdown_and_inline_citation_do_not_lower_support():
    review = verify_answer_claims(
        "D’après le **contexte fourni**, le **3730** possède déjà un **vernis** [1].",
        [_block("Le 3730 possède déjà un vernis.")],
        cited_source_indices=[1],
        use_nli=False,
    )
    assert _statuses(review) == [ClaimStatus.SUPPORTED]
    assert review.claims[0].citation_status == CitationStatus.MATCHED


def test_prudent_absence_for_neighbor_product_is_supported_not_hallucinated():
    review = verify_answer_claims(
        "Aucune information concernant le positionneur 3725 n’est disponible dans ce contexte.",
        [_block("La tropicalisation du Trovis 3730 n’est plus possible.")],
        use_nli=False,
    )
    assert _statuses(review) == [ClaimStatus.SUPPORTED]
    assert review.caveat_required is False


def test_genuinely_partial_claim_remains_partial():
    review = verify_answer_claims(
        "Le 3730 possède un circuit imprimé verni et un boîtier en titane.",
        [_block("Le 3730 possède un circuit imprimé verni.")],
        use_nli=False,
    )
    assert _statuses(review) == [ClaimStatus.PARTIALLY_SUPPORTED]


def test_all_claims_use_one_batched_nli_invocation():
    class FakeNLI:
        def __init__(self):
            self.calls = 0
            self.model = type("Model", (), {"name_or_path": "fake-nli"})()

        def __call__(self, inputs, **_kwargs):
            self.calls += 1
            assert len(inputs) == 2
            return [
                {"label": "ENTAILMENT", "score": 0.91},
                {"label": "CONTRADICTION", "score": 0.88},
            ]

    model = FakeNLI()
    review = verify_answer_claims(
        "Le produit est certifié. Le produit est interdit.",
        [_block("La documentation produit est disponible.")],
        use_nli=True,
        nli_model=model,
    )
    assert model.calls == 1
    assert review.model_calls == 1
    assert _statuses(review) == [ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED]
