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

from api.multi_query_retrieval import build_retrieval_queries, reciprocal_rank_fusion  # noqa: E402


def _chunk(uid, document):
    return {"chunk_uid": uid, "document_id": document, "chunk_id": 0, "text": document}


def test_original_query_is_always_retained_alongside_orchestrator_rewrite():
    queries = build_retrieval_queries(
        original_query="Do I finally have permission to leave the country?",
        orchestrator_query="determine the current status of a travel request",
    )
    assert queries[0][0] == "original"
    assert any(kind == "orchestrator" for kind, _ in queries)
    assert len(queries) <= 3


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
    assert [kind for kind, _ in queries] == ["original"]
