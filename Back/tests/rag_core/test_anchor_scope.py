import numpy as np

from rag_core.anchor_scope import (
    select_anchor_scope,
    select_anchor_scope_adaptive,
    should_expand_scope,
)


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.pair_count = 0

    def predict(self, pairs):
        self.pair_count += len(pairs)
        return np.asarray([self.scores[text] for _, text in pairs], dtype=np.float32)


def meta(uid, document, section, order, text, *, blocks=None, source="file", document_metadata=None):
    return {
        "chunk_uid": uid,
        "document_id": document,
        "section_id": section,
        "page": None,
        "order": order,
        "chunk_id": order,
        "block_ids": blocks or [f"b-{uid}"],
        "previous_chunk_uid": None,
        "next_chunk_uid": None,
        "source": source,
        "document_metadata": document_metadata or {},
        "text": text,
    }


def run_scope(metas, baseline_ids, baseline_scores, relevance, **kwargs):
    texts = [item["text"] for item in metas]
    embeddings = np.asarray([[1.0, 0.0] for _ in metas], dtype=np.float32)
    signals = {
        idx: {"reranker_score": score, "exact_match_bonus": 0.0}
        for idx, score in zip(baseline_ids, baseline_scores)
    }
    return select_anchor_scope(
        query="target evidence",
        query_vec=np.asarray([1.0, 0.0], dtype=np.float32),
        metas=metas,
        texts=texts,
        embeddings=embeddings,
        bm25_scores=np.zeros(len(metas), dtype=np.float32),
        baseline_ids=baseline_ids,
        candidate_signals=signals,
        cross_encoder=FakeCrossEncoder(relevance),
        max_context_chars=kwargs.get("max_context_chars", 12_000),
        final_k=kwargs.get("final_k", 10),
    )


def test_same_native_block_is_kept_but_irrelevant_neighbor_is_rejected():
    rows = [
        meta("a", "doc-1", "s1", 0, "anchor", blocks=["shared"]),
        meta("context", "doc-1", "s1", 1, "preceding context", blocks=["shared"]),
        meta("noise", "doc-1", "s1", 2, "unrelated neighbor"),
    ]
    selected, trace = run_scope(
        rows, [0], [0.9],
        {"anchor": 0.9, "preceding context": 0.1, "unrelated neighbor": 0.05},
    )

    assert [idx for _, idx in selected] == [0, 1]
    decisions = {item["chunk_uid"]: item for item in trace["scope_candidates"]}
    assert decisions["context"]["reason"] == "same_native_unit"
    assert decisions["noise"]["reason"] == "query_relevance_below_anchor_median"


def test_relevant_second_chunk_in_same_section_is_added():
    rows = [
        meta("a", "doc-1", "s1", 0, "anchor"),
        meta("evidence", "doc-1", "s1", 1, "second evidence"),
    ]
    selected, trace = run_scope(
        rows, [0], [0.8], {"anchor": 0.8, "second evidence": 0.9}
    )

    assert [idx for _, idx in selected] == [0, 1]
    assert trace["multi_chunk_scope"] is True


def test_multiple_non_redundant_anchor_zones_are_preserved():
    rows = [
        meta("a", "doc-1", "s1", 0, "first anchor"),
        meta("duplicate-zone", "doc-1", "s1", 1, "duplicate zone"),
        meta("b", "doc-1", "s2", 2, "second section"),
        meta("c", "doc-2", "s1", 0, "second document"),
    ]
    selected, trace = run_scope(
        rows, [0, 1, 2, 3], [0.9, 0.88, 0.88, 0.88],
        {"first anchor": 0.9, "duplicate zone": 0.88, "second section": 0.88, "second document": 0.88},
    )

    assert trace["selected_anchor_ids"] == [0, 2, 3]
    assert trace["multi_anchor"] is True
    duplicate = next(item for item in trace["anchors"] if item["chunk_uid"] == "duplicate-zone")
    assert duplicate["reason"] == "redundant_zone"
    assert {idx for _, idx in selected} >= {0, 2, 3}


def test_budget_keeps_anchor_and_traces_discarded_scope():
    rows = [
        meta("a", "doc-1", "s1", 0, "anchor text"),
        meta("b", "doc-1", "s1", 1, "relevant expansion"),
    ]
    selected, trace = run_scope(
        rows, [0], [0.8], {"anchor text": 0.8, "relevant expansion": 0.9},
        max_context_chars=len("anchor text"),
    )

    assert [idx for _, idx in selected] == [0]
    assert trace["discarded_by_budget"][0]["chunk_uid"] == "b"
    assert trace["discarded_by_budget"][0]["reason"] == "character_budget"


def test_linked_email_attachment_can_enter_scope_when_relevant():
    rows = [
        meta(
            "mail", "mail-doc", None, 0, "email anchor", source="email",
            document_metadata={
                "thread_id": "thread-1",
                "attachments": [{"document_id": "attachment-doc"}],
            },
        ),
        meta("attachment", "attachment-doc", "invoice", 0, "attachment proof"),
    ]
    selected, trace = run_scope(
        rows, [0], [0.8], {"email anchor": 0.8, "attachment proof": 0.9}
    )

    assert [idx for _, idx in selected] == [0, 1]
    assert trace["scope_candidates"][0]["relation"] == "linked_attachment"


def test_scope_can_remain_anchor_only():
    rows = [
        meta("a", "doc-1", "s1", 0, "strong anchor"),
        meta("b", "doc-1", "s1", 1, "weak aside"),
    ]
    selected, trace = run_scope(
        rows, [0], [0.95], {"strong anchor": 0.95, "weak aside": 0.1}
    )

    assert [idx for _, idx in selected] == [0]
    assert trace["multi_chunk_scope"] is False


def test_adaptive_skips_strong_multi_evidence_result():
    rows = [
        meta("a", "doc-1", None, 0, "strong proof", source="email"),
        meta("b", "doc-2", None, 0, "other proof", source="email"),
    ]
    expand, decision = should_expand_scope(
        query="Compare les deux résultats",
        baseline_ids=[0, 1],
        candidate_signals={
            0: {"reranker_score": 0.91},
            1: {"reranker_score": 0.88},
        },
        metas=rows,
    )

    assert expand is False
    assert decision["reason"] == "strong_initial_evidence"


def test_adaptive_skips_weak_single_fact_and_does_not_rerank():
    rows = [
        meta("a", "doc-1", None, 0, "weak result", source="email"),
        meta("b", "doc-2", None, 0, "noise", source="email"),
    ]
    cross_encoder = FakeCrossEncoder({})
    selected, trace = select_anchor_scope_adaptive(
        query="Quel est le budget marketing annuel approuvé pour 2027 ?",
        query_vec=np.asarray([1.0, 0.0], dtype=np.float32),
        metas=rows,
        texts=[row["text"] for row in rows],
        embeddings=np.asarray([[1.0, 0.0] for _ in rows], dtype=np.float32),
        bm25_scores=np.zeros(len(rows), dtype=np.float32),
        baseline_ids=[0, 1],
        candidate_signals={
            0: {"reranker_score": 0.002},
            1: {"reranker_score": 0.001},
        },
        cross_encoder=cross_encoder,
    )

    assert [idx for _, idx in selected] == [0, 1]
    assert trace["scope_triggered"] is False
    assert trace["rescored_candidates"] == 0
    assert cross_encoder.pair_count == 0


def test_adaptive_dashboard_style_query_keeps_one_anchor_and_recovers_thread():
    thread = {"thread_id": "dashboard-thread", "chronological_key": "2026-06-01"}
    rows = [
        meta("reply", "reply-doc", None, 0, "confirmed cause", source="email", document_metadata=thread),
        meta("reply-body", "reply-doc", None, 1, "staging result", source="email", document_metadata=thread),
        meta(
            "original", "original-doc", None, 0, "42 million rows",
            source="email",
            document_metadata={"thread_id": "dashboard-thread", "chronological_key": "2026-05-31"},
        ),
        meta("noise", "noise-doc", None, 0, "unrelated", source="email"),
    ]
    cross_encoder = FakeCrossEncoder({
        "staging result": 0.004,
        "42 million rows": 0.006,
    })
    selected, trace = select_anchor_scope_adaptive(
        query="Combien de lignes, quelle cause et quel résultat après correction ?",
        query_vec=np.asarray([1.0, 0.0], dtype=np.float32),
        metas=rows,
        texts=[row["text"] for row in rows],
        embeddings=np.asarray([[1.0, 0.0] for _ in rows], dtype=np.float32),
        bm25_scores=np.asarray([0.9, 0.8, 0.85, 0.0], dtype=np.float32),
        baseline_ids=[0, 3],
        candidate_signals={
            0: {"reranker_score": 0.011},
            3: {"reranker_score": 0.0001},
        },
        cross_encoder=cross_encoder,
        final_k=4,
    )

    selected_ids = [idx for _, idx in selected]
    assert trace["scope_triggered"] is True
    assert trace["selected_anchor_ids"] == [0]
    assert trace["multi_anchor"] is False
    assert 2 in selected_ids
    assert any(
        item["chunk_uid"] == "original" and item["relation"] == "same_thread"
        for item in trace["scope_candidates"] if item["selected"]
    )
    assert trace["rescored_candidates"] <= 2


def test_adaptive_uses_second_zone_only_for_explicit_comparison():
    rows = [
        meta("a", "doc-1", "s1", 0, "first system"),
        meta("b", "doc-2", "s1", 0, "second system"),
        meta("c", "doc-3", "s1", 0, "noise"),
    ]
    selected, trace = select_anchor_scope_adaptive(
        query="Compare les évolutions des deux systèmes",
        query_vec=np.asarray([1.0, 0.0], dtype=np.float32),
        metas=rows,
        texts=[row["text"] for row in rows],
        embeddings=np.asarray([[1.0, 0.0] for _ in rows], dtype=np.float32),
        bm25_scores=np.zeros(len(rows), dtype=np.float32),
        baseline_ids=[0, 1, 2],
        candidate_signals={
            0: {"reranker_score": 0.02},
            1: {"reranker_score": 0.01},
            2: {"reranker_score": 0.001},
        },
        cross_encoder=None,
    )

    assert trace["selected_anchor_ids"] == [0, 1]
    assert trace["multi_anchor"] is True
    assert {idx for _, idx in selected} >= {0, 1}
