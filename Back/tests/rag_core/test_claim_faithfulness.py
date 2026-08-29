from rag_core.faithfulness import (
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
            _block("Le 3730 possède un circuit imprimé verni.", "right:c1"),
            _block("Le 3725 utilise un boîtier en aluminium.", "wrong:c2"),
        ],
        cited_source_indices=[2],
        use_nli=False,
    )
    assert _statuses(review) == [ClaimStatus.SUPPORTED]
    assert review.claims[0].citation_correct is False
    assert review.caveat_required is True


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
