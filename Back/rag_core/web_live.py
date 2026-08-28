# -*- coding: utf-8 -*-
"""
Recherche web live (Tavily) — inchangé.
"""
import requests
from .constants import TAVILY_API_KEY
from runtime_settings import get_runtime_settings

def web_search_context(
    query: str,
    max_chars: int | None = None,
    k: int | None = None,
) -> str:
    settings = get_runtime_settings()
    if not settings.features.enable_web_search or not TAVILY_API_KEY:
        return ""
    max_chars = settings.conversation.web_max_chars if max_chars is None else max_chars
    k = settings.conversation.web_result_k if k is None else k
    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": k,
            "search_depth": "basic",
            "include_answer": False,
        }
        r = requests.post(
            url,
            json=payload,
            timeout=settings.timeouts.web_http_sec,
        )
        r.raise_for_status()
        data = r.json()
        parts = []
        for item in data.get("results", []):
            title = item.get("title", "")
            link = item.get("url", "")
            snippet = item.get("content") or item.get("snippet") or ""
            if title or snippet:
                parts.append(f"[WEB] {title}\n{link}\n{snippet}")
        txt = "\n\n".join(parts).strip()
        return txt[:max_chars]
    except Exception:
        return ""
