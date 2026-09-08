from unittest.mock import patch

from rag_core.llm import ask_mistral_with_context
from rag_core.llm_stream import ask_mistral_with_context_stream


CONTEXT = "[1] Une action a été déposée le 4 avril. [2] Aucune décision finale n'est documentée."


def _system_message(payload):
    return payload["messages"][0]["content"]


def test_sync_strict_prompt_requires_facts_before_a_caveat_for_direct_evidence():
    captured = {}

    with patch("rag_core.llm.generate_from_payload", side_effect=lambda payload: captured.update(payload) or "ok"):
        assert ask_mistral_with_context(
            "La demande a-t-elle été acceptée ?", CONTEXT, history=[], evidence_mode="direct"
        ) == "ok"

    prompt = _system_message(captured)
    assert "ne réponds jamais uniquement par une abstention" in prompt
    assert "faits pertinents établis" in prompt
    assert "ce qui manque pour conclure" in prompt
    assert "preuve exploitable" in prompt
    assert 'réponds exactement : "Je ne sais pas"' not in prompt


def test_sync_general_prompt_distinguishes_missing_results_from_source_access():
    captured = {}

    with patch("rag_core.llm.generate_from_payload", side_effect=lambda payload: captured.update(payload) or "ok"):
        assert ask_mistral_with_context("Question factuelle", "", history=[]) == "ok"

    prompt = _system_message(captured)
    assert "sources locales indexées" in prompt
    assert "absence d'accès" in prompt


def test_streaming_general_prompt_uses_the_same_local_source_capability_contract():
    sync_captured = {}
    stream_captured = {}

    with patch("rag_core.llm.generate_from_payload", side_effect=lambda payload: sync_captured.update(payload) or "ok"):
        assert ask_mistral_with_context("Question factuelle", "", history=[]) == "ok"
    with patch("rag_core.llm_stream.stream_from_payload", side_effect=lambda payload: stream_captured.update(payload) or iter(["ok"])):
        assert list(ask_mistral_with_context_stream("Question factuelle", "", history=[])) == ["ok"]

    sync_prompt = _system_message(sync_captured)
    stream_prompt = _system_message(stream_captured)
    capability = "Auxilium peut rechercher dans ses sources locales indexées"
    assert capability in sync_prompt
    assert capability in stream_prompt
    assert "absence d'accès" in sync_prompt
    assert "absence d'accès" in stream_prompt


def test_streaming_strict_prompt_keeps_the_same_partial_evidence_contract():
    captured = {}

    with patch("rag_core.llm_stream.stream_from_payload", side_effect=lambda payload: captured.update(payload) or iter(["ok"])):
        assert list(ask_mistral_with_context_stream(
            "Où en est le dossier ?", CONTEXT, history=[], evidence_mode="related"
        )) == ["ok"]

    prompt = _system_message(captured)
    assert "ne réponds jamais uniquement par une abstention" in prompt
    assert "ordre chronologique" in prompt
    assert "related evidence" in prompt
    assert 'réponds exactement : "Je ne sais pas"' not in prompt


def test_general_complement_keeps_internal_facts_grounded_and_uncited():
    captured = {}
    with patch("rag_core.llm.generate_from_payload", side_effect=lambda payload: captured.update(payload) or "ok"):
        assert ask_mistral_with_context(
            "Quel est le prix et a quoi sert ce produit ?", CONTEXT, history=[],
            evidence_mode="direct", allow_general_complement=True,
        ) == "ok"

    prompt = _system_message(captured)
    assert "Exception multi-source" in prompt
    assert "faits propres a l'entreprise" in prompt
    assert "ne lui attribue aucune citation documentaire" in prompt
