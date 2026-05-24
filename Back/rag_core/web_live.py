# -*- coding: utf-8 -*-
"""
Recherche web live (Tavily) — inchangé.
"""
import requests
from .constants import ENABLE_WEB_SEARCH, TAVILY_API_KEY

def web_search_context(query: str, max_chars: int = 4000, k: int = 5) -> str:
    if not ENABLE_WEB_SEARCH or not TAVILY_API_KEY:
        return ""
    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": k,
            "search_depth": "basic",
            "include_answer": False,
        }
        r = requests.post(url, json=payload, timeout=30)
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
