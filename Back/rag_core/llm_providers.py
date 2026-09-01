"""Provider boundary for text generation; prompt construction stays in ``llm``."""
from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Iterator, Sequence

from runtime_settings import GenerationSettings, get_runtime_settings


def _temporary_generation_diagnostic(*, provider: str, request_args: dict[str, Any], stream: bool) -> None:
    """Emit correlation-safe, temporary diagnostics without logging prompt text."""
    try:
        # Imported lazily to keep rag_core usable outside the FastAPI package.
        from api.diagnostics import current_request_id, current_stage, log_event
        request_id = current_request_id()
        if not request_id:
            return
        serialized = json.dumps(request_args, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        log_event(
            request_id,
            "provider_call_effective",
            provider=provider,
            stage=current_stage(),
            stream=stream,
            model=request_args.get("model"),
            prompt_hash=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            temperature=request_args.get("temperature"),
            top_p=request_args.get("top_p"),
            seed=request_args.get("seed"),
            reasoning_effort=(request_args.get("reasoning") or {}).get("effort"),
        )
    except Exception:
        # Diagnostics must never alter generation availability.
        pass


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: Sequence[dict[str, str]], *, model: str, temperature: float,
                 top_p: float, max_tokens: int, settings: GenerationSettings) -> str: ...

    @abstractmethod
    def stream(self, messages: Sequence[dict[str, str]], *, model: str, temperature: float,
               top_p: float, max_tokens: int, settings: GenerationSettings) -> Iterator[str]: ...


class MistralProvider(LLMProvider):
    def _request(self, messages, *, model, temperature, top_p, max_tokens, settings, stream=False):
        import requests
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY absente. Definis-la puis relance l'API.")
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        payload = {"model": model, "messages": list(messages), "temperature": temperature,
                   "top_p": top_p, "max_tokens": max_tokens}
        if stream:
            payload["stream"] = True
        _temporary_generation_diagnostic(provider="mistral", request_args=payload, stream=stream)
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=settings.http_timeout_sec,
            stream=stream,
        )
        if response.status_code == 401:
            raise RuntimeError("401 Unauthorized: verifie MISTRAL_API_KEY.")
        if response.status_code == 429:
            raise RuntimeError("429 Too Many Requests: quota/ratelimit.")
        response.raise_for_status()
        return response

    def generate(self, messages, **kwargs) -> str:
        return self._request(messages, stream=False, **kwargs).json()["choices"][0]["message"]["content"]

    def stream(self, messages, **kwargs) -> Iterator[str]:
        import json
        with self._request(messages, stream=True, **kwargs) as response:
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {}).get("content", "")
                except (ValueError, KeyError, IndexError, TypeError):
                    continue
                if delta:
                    yield delta


class OpenAIProvider(LLMProvider):
    def _client(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY absente. Definis-la puis relance l'API.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Le SDK officiel openai est requis (pip install openai).") from exc
        return OpenAI(api_key=api_key)

    @staticmethod
    def _request_args(messages, *, model, max_tokens, settings, **_ignored):
        # Responses separates system instructions from the conversation input.
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        instructions = "\n\n".join(m["content"] for m in messages if m.get("role") == "system") or None
        input_items = [
            {"role": m["role"], "content": m["content"]}
            for m in messages if m.get("role") != "system"
        ]
        args: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": settings.reasoning_effort},
        }
        if instructions:
            args["instructions"] = instructions
        # GPT-5.6 Responses does not receive Mistral's temperature/top_p blindly.
        return args

    def generate(self, messages, **kwargs) -> str:
        args = self._request_args(messages, **kwargs)
        _temporary_generation_diagnostic(provider="openai", request_args=args, stream=False)
        response = self._client().responses.create(**args)
        return response.output_text

    def stream(self, messages, **kwargs) -> Iterator[str]:
        args = self._request_args(messages, **kwargs)
        _temporary_generation_diagnostic(provider="openai", request_args=args, stream=True)
        stream = self._client().responses.create(stream=True, **args)
        for event in stream:
            if getattr(event, "type", None) == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield delta


def get_llm_provider(provider_name: str | None = None) -> LLMProvider:
    provider_name = provider_name or get_runtime_settings().generation.provider
    if provider_name == "mistral":
        return MistralProvider()
    if provider_name == "openai":
        return OpenAIProvider()
    raise RuntimeError(f"Provider LLM non supporte: {provider_name}")


def generate(messages: Sequence[dict[str, str]], **kwargs) -> str:
    return get_llm_provider().generate(messages, **kwargs)


def stream(messages: Sequence[dict[str, str]], **kwargs) -> Iterator[str]:
    yield from get_llm_provider().stream(messages, **kwargs)


def generate_from_payload(payload: dict[str, Any]) -> str:
    """Compatibility adapter while the existing prompt builders retain their payload shape."""
    data = dict(payload)
    return generate(data.pop("messages"), settings=get_runtime_settings().generation, **data)


def stream_from_payload(payload: dict[str, Any]) -> Iterator[str]:
    data = dict(payload)
    data.pop("stream", None)
    yield from stream(data.pop("messages"), settings=get_runtime_settings().generation, **data)
