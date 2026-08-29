import os
from unittest.mock import patch

import pytest

from rag_core.llm_providers import OpenAIProvider, generate_from_payload, stream_from_payload
from runtime_settings import GenerationSettings, load_runtime_settings


class _Response:
    output_text = "OK"


class _Event:
    def __init__(self, event_type, delta=""):
        self.type = event_type
        self.delta = delta


class _Responses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([_Event("response.output_text.delta", "Bon"), _Event("response.output_text.delta", "jour")])
        return _Response()


class _Client:
    def __init__(self):
        self.responses = _Responses()


def _settings():
    return GenerationSettings(provider="openai", model="gpt-5.6-luna", reasoning_effort="medium")


def test_openai_non_streaming_uses_responses_api_with_existing_messages():
    client = _Client()
    provider = OpenAIProvider()
    with patch.object(provider, "_client", return_value=client):
        answer = provider.generate(
            [{"role": "system", "content": "System prompt"}, {"role": "user", "content": "Question"}],
            model="gpt-5.6-terra", temperature=0.6, top_p=0.9, max_tokens=42, settings=_settings(),
        )
    assert answer == "OK"
    call = client.responses.calls[0]
    assert call["model"] == "gpt-5.6-terra"
    assert call["instructions"] == "System prompt"
    assert call["input"] == [{"role": "user", "content": "Question"}]
    assert call["max_output_tokens"] == 42
    assert call["reasoning"] == {"effort": "medium"}
    assert "temperature" not in call and "top_p" not in call


def test_openai_streaming_yields_each_delta_without_buffering():
    client = _Client()
    provider = OpenAIProvider()
    with patch.object(provider, "_client", return_value=client):
        chunks = list(provider.stream(
            [{"role": "user", "content": "Question"}], model="gpt-5.6-sol",
            temperature=0.6, top_p=0.9, max_tokens=42, settings=_settings(),
        ))
    assert chunks == ["Bon", "jour"]
    assert client.responses.calls[0]["stream"] is True


def test_openai_missing_key_is_explicit(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY absente"):
        OpenAIProvider()._client()


def test_openai_provider_and_model_are_accepted_by_runtime_settings():
    settings = load_runtime_settings(
        config_data={"llm": {"provider": "openai", "model": "gpt-5.6-sol"}}, env={}
    )
    assert settings.generation.provider == "openai"
    assert settings.generation.model == "gpt-5.6-sol"


def test_payload_adapters_pass_generation_settings_not_runtime_settings():
    runtime = type("Runtime", (), {"generation": _settings()})()

    class Provider:
        def generate(self, _messages, **kwargs):
            assert kwargs["settings"] is runtime.generation
            return "OK"

        def stream(self, _messages, **kwargs):
            assert kwargs["settings"] is runtime.generation
            yield "OK"

    with (
        patch("rag_core.llm_providers.get_runtime_settings", return_value=runtime),
        patch("rag_core.llm_providers.get_llm_provider", return_value=Provider()),
    ):
        assert generate_from_payload({"messages": [], "model": "gpt-5.6-luna", "temperature": 0.6, "top_p": 0.9, "max_tokens": 1}) == "OK"
        assert list(stream_from_payload({"messages": [], "model": "gpt-5.6-luna", "temperature": 0.6, "top_p": 0.9, "max_tokens": 1, "stream": True})) == ["OK"]
