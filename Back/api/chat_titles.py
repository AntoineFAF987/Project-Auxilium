"""Small, conversation-only LLM helper for automatic chat titles."""
from __future__ import annotations

import re
from typing import Iterable, Mapping

from rag_core.llm_providers import get_llm_provider
from runtime_settings import get_runtime_settings


_TITLE_SYSTEM_PROMPT = """Analyse la conversation suivante et génère un titre court et précis qui résume son sujet principal.

Contraintes :
- 3 à 8 mots idéalement, jamais plus de 10
- pas de guillemets, de point final, de préfixe comme « Titre : », ni de markdown
- conserve les noms de produits, entreprises et termes techniques importants
- utilise la langue principale de la conversation
- retourne uniquement le titre"""


def _conversation_context(messages: Iterable[Mapping[str, str]]) -> str:
    """Keep only the first exchanges; titles do not need retrieval or long history."""
    parts: list[str] = []
    for message in list(messages)[:4]:
        role = "Utilisateur" if message.get("role") == "user" else "Auxilium"
        content = re.sub(r"\s+", " ", str(message.get("content") or "")).strip()
        if content:
            parts.append(f"{role} : {content[:900]}")
    return "\n".join(parts)[:2800]


def _clean_title(value: str) -> str:
    title = re.sub(r"\s+", " ", value or "").strip()
    title = re.sub(r"(?i)^\s*titre\s*:\s*", "", title)
    title = title.strip(" \t\n\r`*_\"'«»")
    title = re.sub(r"[.!?]+$", "", title).strip()
    words = title.split()
    return " ".join(words[:10])[:90].strip()


def generate_chat_title(messages: Iterable[Mapping[str, str]]) -> str | None:
    """Generate a compact title using the configured LLM, without RAG or web access."""
    context = _conversation_context(messages)
    if not context:
        return None
    runtime = get_runtime_settings()
    settings = runtime.generation.model_copy(update={"reasoning_effort": "low"})
    response = get_llm_provider(settings.provider).generate(
        [
            {"role": "system", "content": _TITLE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Conversation :\n{context}"},
        ],
        model=settings.model,
        temperature=0.25,
        top_p=0.9,
        max_tokens=32,
        settings=settings,
    )
    return _clean_title(response) or None
