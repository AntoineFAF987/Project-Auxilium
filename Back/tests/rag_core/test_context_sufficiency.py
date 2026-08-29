from rag_core.context_sufficiency import (
    SufficiencyPolicy,
    select_adaptive_context,
)
from eval.run_current import is_premature_stop


def _run(query, texts, *, scores=None, documents=None, protected=()):
    scores = scores or [0.9] + [0.2] * (len(texts) - 1)
    documents = documents or [f"doc-{index}" for index in range(len(texts))]
    metas = [
        {"document_id": document, "chunk_uid": f"chunk-{index}"}
        for index, document in enumerate(documents)
    ]
    signals = {
        index: {
            "reranker_score": score,
            "dense_rank": index + 1,
            "bm25_rank": index + 1,
            "exact_match_bonus": 0.0,
        }
        for index, score in enumerate(scores)
    }
    return select_adaptive_context(
        query=query,
        candidates=[(score, index) for index, score in enumerate(scores)],
        metas=metas,
        texts=texts,
        candidate_signals=signals,
        protected_candidate_ids=protected,
        final_k=10,
    )


def test_strong_explicit_single_evidence_stops_at_one():
    selected, trace = _run(
        "Quel est le matériau du corps de cette vanne ?",
        ["Le matériau du corps de cette vanne est l'acier inoxydable.", "Autre passage."],
    )
    assert [idx for _, idx in selected] == [0]
    assert trace["final_decision"]["reason"] == "strong_coherent_single_evidence"


def test_weak_top_adds_more_evidence():
    selected, _ = _run(
        "Quel matériau compose le corps de la vanne ?",
        ["Notice générale.", "Le corps de la vanne est en bronze."],
        scores=[0.1, 0.2],
    )
    assert [idx for _, idx in selected] == [0, 1]


def test_comparison_requires_two_distinct_documents():
    selected, trace = _run(
        "Compare A et B sur la pression et les matériaux.",
        ["A: pression 10 bar, acier.", "B: pression 12 bar, bronze.", "Divers."],
        documents=["a", "b", "c"],
    )
    assert [idx for _, idx in selected] == [0, 1]
    assert trace["intent"]["kind"] == "comparison"


def test_two_document_question_keeps_protected_anchor_scope_evidence():
    selected, trace = _run(
        "Quelle était la date initiale et quel nouveau calendrier a été décidé ?",
        ["Date initiale: 12 août.", "Nouveau calendrier: 5 septembre."],
        documents=["original", "reply"],
        protected=(0, 1),
    )
    assert [idx for _, idx in selected] == [0, 1]
    assert trace["final_decision"]["protected_evidence_missing"] == 0


def test_redundant_candidate_is_skipped():
    duplicate = "La vanne supporte 10 bar et son corps est en acier inoxydable."
    selected, trace = _run(
        "Quelle pression et quel matériau pour la vanne ?",
        [duplicate, duplicate, "La température maximale est 180 degrés."],
        scores=[0.2, 0.2, 0.2],
    )
    assert 1 not in [idx for _, idx in selected]
    assert trace["skipped_redundant"][0]["candidate_id"] == 1


def test_pilot_dashboard_investigation_preserves_two_required_proofs():
    selected, _ = _run(
        "Pour les lenteurs du dashboard, combien de lignes étaient concernées, quelle cause a été confirmée et quel résultat a été mesuré après correction ?",
        [
            "customer_events contenait 42 millions de lignes.",
            "Le scan séquentiel a été corrigé par un index; 8,3 s est devenu 740 ms.",
            "Information sans rapport.",
        ],
        documents=["initial", "reply", "noise"],
        protected=(0, 1),
    )
    assert {idx for _, idx in selected} >= {0, 1}


def test_unanswerable_weak_query_does_not_fill_to_ten():
    selected, trace = _run(
        "Quel est le budget marketing annuel approuvé pour 2027 ?",
        [f"Passage sans rapport numéro {index}." for index in range(10)],
        scores=[0.05] * 10,
    )
    assert len(selected) == 3
    assert trace["final_decision"]["reason"] == "weak_evidence_patience_exhausted"


def test_strong_but_incomplete_signal_reaches_gold_at_rank_five():
    selected, _ = _run(
        "Quel matériau cryogénique compose le joint spécial ?",
        [
            "Notice de maintenance.", "Consignes générales.", "Pression nominale.",
            "Température ambiante.", "Le joint spécial cryogénique est en PTFE.",
        ],
        scores=[0.49, 0.3, 0.2, 0.15, 0.1],
    )
    assert 4 in [idx for _, idx in selected]


def test_exhaustive_question_does_not_use_weak_evidence_patience():
    selected, _ = _run(
        "Énumère tous les avantages et limites du compte.",
        [
            "Présentation générale.", "Autre introduction.", "Navigation.",
            "Les avantages sont la synchronisation; la limite dépend du marché.",
        ],
        scores=[0.05] * 4,
    )
    assert 3 in [idx for _, idx in selected]


def test_premature_stop_detects_available_but_omitted_documentary_gold():
    premature, eligible = is_premature_stop(
        selected_ids=["document::a"],
        available_ids=["document::a", "document::gold"],
        gold_ids=["document::gold"],
    )
    assert eligible is True
    assert premature is True


def test_thresholds_are_explicit_and_configurable():
    policy = SufficiencyPolicy(strong_reranker_score=0.7, weak_evidence_patience=4)
    assert policy.strong_reranker_score == 0.7
    assert policy.weak_evidence_patience == 4
