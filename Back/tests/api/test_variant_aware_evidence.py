import sys
import types
from pathlib import Path
from unittest.mock import Mock


_BACK_ROOT = Path(__file__).resolve().parents[2]
if "api" not in sys.modules:
    package = types.ModuleType("api")
    package.__path__ = [str(_BACK_ROOT / "api")]
    sys.modules["api"] = package

stub = types.ModuleType("api.index_singleton")
stub.idx = object()
stub.format_context_for_llm = Mock()
stub.fuse_contiguous_passages = Mock()
stub.clip_context_blocks = Mock()
stub.ask_mistral_with_context = Mock()
stub.answerability_guard = Mock()
stub.keyword_overlap_count = lambda query, context: len(set(query.casefold().split()) & set(context.casefold().split()))
stub.trim_history = lambda rows, max_turns: list(rows)
stub.classify_smalltalk_semantic = Mock(return_value="")
stub.web_search_context = Mock()
stub.RETRIEVE_K = 12
stub.TOP_K_FAISS = 24
stub.HYBRID_ALPHA = 0.65
stub.FUSE_ADJACENT_GAP = 1
stub.MAX_CONTEXT_CHARS = 12000
stub.FINAL_K = 6
sys.modules.setdefault("api.index_singleton", stub)

from api.answer_pipeline import (  # noqa: E402
    _annotate_variant_aware_candidates, _check_context_relevance, _evaluate_evidence,
    _evidence_selection_order, _final_information_need_coverage,
    _protect_answer_bearing_fused_blocks, _documentary_fallback_query,
)


_FR = "Quelle est la bande morte des actionneurs PS AMS ?"
_EN = "PS AMS actuator dead band"
_VARIANTS = [("orchestrator", "bande morte des actionneurs PS AMS"), ("cross_language", _EN)]


def _candidate(text, *, rank=1, score=0.99, retrieved_by="cross_language"):
    return {
        "chunk_uid": text[:12], "document_id": text[:12], "text": text,
        "retrieved_by": [retrieved_by],
        "per_query_rank": {retrieved_by: rank},
        "per_query_score": {retrieved_by: score},
    }


def test_english_direct_chunk_is_relevant_via_equivalent_query_variant():
    chunk = _candidate("The dead band of PS-AMS actuators is adjustable between 0.5% and 5%.")
    annotated = _annotate_variant_aware_candidates([(0.016, chunk)], _VARIANTS)
    meta = annotated[0][1]
    assert meta["best_matching_query_type"] == "cross_language"
    assert meta["best_query_term_coverage"] >= 0.75
    assert meta["variant_aware_promotion_applied"] is True
    assert _check_context_relevance(_FR, meta["text"], query_variants=_VARIANTS) is True
    decision = _evaluate_evidence(_FR, annotated, guard_ok=True, context_is_relevant=True, overlap=3, query_variants=_VARIANTS)
    assert decision.mode == "direct"


def test_strong_single_variant_direct_chunk_can_be_promoted_over_repeated_overview():
    overview = _candidate("PS-AMS actuator product overview.", rank=1, score=0.8, retrieved_by="orchestrator")
    overview["retrieved_by"] = ["original_autonomous", "orchestrator", "cross_language"]
    overview["per_query_rank"] = {"original_autonomous": 1, "orchestrator": 1, "cross_language": 2}
    overview["per_query_score"] = {"original_autonomous": 0.8, "orchestrator": 0.8, "cross_language": 0.7}
    direct = _candidate("The dead band of PS-AMS actuators is 0.5% to 5%.", rank=1, score=0.99)
    rows = _annotate_variant_aware_candidates([(0.047, overview), (0.016, direct)], _VARIANTS)
    by_text = {meta["text"]: meta for _score, meta in rows}
    assert by_text[direct["text"]]["variant_aware_promotion_applied"] is True
    assert by_text[overview["text"]]["variant_aware_promotion_applied"] is False
    # RRF scores remain untouched, but bounded evidence selection retains the
    # evaluated answer-bearing candidate before the repeated overview.
    assert _evidence_selection_order(rows)[0][1]["text"] == direct["text"]


def test_cross_language_product_overview_is_not_promoted_without_requested_predicate():
    chunk = _candidate("HV02 general product introduction.")
    rows = _annotate_variant_aware_candidates([(0.016, chunk)], [("cross_language", "HV02 operating temperature")])
    assert rows[0][1]["variant_aware_promotion_applied"] is False


def test_french_variant_remains_eligible_when_it_is_the_best_match():
    chunk = _candidate("La bande morte des actionneurs PS AMS est réglable.", retrieved_by="orchestrator")
    rows = _annotate_variant_aware_candidates([(0.016, chunk)], _VARIANTS)
    assert rows[0][1]["best_matching_query_type"] == "orchestrator"


def test_topic_only_candidate_is_not_answer_bearing_for_dead_band_need():
    generic = _candidate("PS AMS intelligent actuator with integrated microcontroller.")
    rows = _annotate_variant_aware_candidates([(0.047, generic)], [("cross_language", _EN)])
    meta = rows[0][1]
    assert meta["answers_information_need"] is False
    assert meta["information_need_coverage"] == "none"


def test_documentary_fallback_unwraps_reply_framing_without_losing_exact_reference():
    normalized, removed = _documentary_fallback_query(
        "Un client me demande par mail si le positionneur 3730 peut être tropicalisé. Je lui réponds quoi ?"
    )
    assert "3730" in normalized
    assert "tropicalisé" in normalized
    assert "Je lui réponds quoi" not in normalized
    assert removed


def test_documentary_fallback_keeps_a_distinct_exact_reference_in_reply_framing():
    normalized, removed = _documentary_fallback_query(
        "Un client me demande par mail si le positionneur 3731 peut être tropicalisé. Je lui réponds quoi ?"
    )
    assert "3731" in normalized
    assert "tropicalisé" in normalized.casefold()
    assert "Je lui réponds quoi" not in normalized
    assert removed


def test_direct_value_is_protected_after_fusion_reorders_by_rrf_score():
    generic = _candidate("PS AMS intelligent actuator product overview.", rank=1, score=0.99)
    direct = _candidate("The dead band of PS-AMS actuators is adjustable between 0.5% and 5%.", rank=1, score=0.98)
    rows = _annotate_variant_aware_candidates([(0.047, generic), (0.016, direct)], [("cross_language", _EN)])
    direct_uid = direct["chunk_uid"]
    # Model the post-fusion order: generic score first, direct answer second.
    fused = [(0.047, {**rows[0][1], "fused_chunk_uids": [generic["chunk_uid"]]}), (0.016, {**rows[1][1], "fused_chunk_uids": [direct_uid]})]
    protected = _protect_answer_bearing_fused_blocks(fused, {direct_uid})
    assert protected[0][1]["text"] == direct["text"]
    status, score, kind = _final_information_need_coverage(protected, [("cross_language", _EN)])
    assert (status, kind) == ("complete", "cross_language")
    assert score == 1.0
