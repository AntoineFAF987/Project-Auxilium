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

from api.multi_query_retrieval import build_retrieval_queries, reciprocal_rank_fusion, resolve_retrieval_query  # noqa: E402


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
