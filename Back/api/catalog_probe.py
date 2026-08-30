"""Low-cost title/metadata probe used before an orchestrated GENERAL answer."""
from __future__ import annotations

from dataclasses import dataclass
import re
import time
import unicodedata
from typing import Any


_GENERIC = {
    "a", "an", "and", "are", "about", "comment", "cest", "ce", "de", "des", "du", "dans",
    "document", "documents", "explain", "est", "for", "how", "information", "in", "is", "la",
    "le", "les", "me", "mes", "my", "of", "on", "projet", "project", "rapport", "report",
    "source", "sources", "the", "to", "une", "un", "what", "with", "works", "fonctionne",
}
_CATALOG_CACHE: dict[int, tuple[int, list[tuple[str | None, str]]]] = {}


def _canonical(value: str) -> str:
    text = unicodedata.normalize("NFKD", value.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _tokens(value: str, *, keep_generic: bool = False) -> set[str]:
    tokens = set(_canonical(value).split())
    return tokens if keep_generic else {token for token in tokens if len(token) >= 3 and token not in _GENERIC}


def _title(meta: dict[str, Any]) -> str:
    document = meta.get("document_metadata") or {}
    return " ".join(str(value or "") for value in (
        meta.get("title"), meta.get("display_name"), meta.get("file"),
        document.get("title"), document.get("display_name"), document.get("filename"),
        document.get("subject"), document.get("file"),
    ))


@dataclass(frozen=True)
class CatalogProbeResult:
    executed: bool
    strong_match: bool
    matched_document_id: str | None = None
    matched_title: str | None = None
    reason: str | None = None
    timing_ms: float = 0.0

    def trace(self) -> dict[str, Any]:
        return {
            "executed": self.executed, "strong_match": self.strong_match,
            "matched_document_id": self.matched_document_id, "matched_title": self.matched_title,
            "reason": self.reason, "timing_ms": self.timing_ms,
        }


def probe_document_catalog(query: str, corpus: list[dict[str, Any]] | None) -> CatalogProbeResult:
    """Inspect only already-loaded document metadata; never call retrieval indexes."""
    started = time.perf_counter()
    if not corpus:
        return CatalogProbeResult(False, False, reason="catalog_unavailable", timing_ms=round((time.perf_counter() - started) * 1000, 3))
    query_tokens = _tokens(query)
    query_title_tokens = _tokens(query, keep_generic=True)
    query_phrase = _canonical(query)
    if not query_tokens:
        return CatalogProbeResult(True, False, reason="no_significant_query_tokens", timing_ms=round((time.perf_counter() - started) * 1000, 3))
    cache_key = id(corpus)
    cached = _CATALOG_CACHE.get(cache_key)
    if cached is None or cached[0] != len(corpus):
        seen_documents: set[str] = set()
        catalog: list[tuple[str | None, str]] = []
        for meta in corpus:
            document_id = str(meta.get("document_id") or "")
            identity = document_id or str(meta.get("file") or meta.get("path") or id(meta))
            if identity in seen_documents:
                continue
            seen_documents.add(identity)
            catalog.append((document_id or None, _title(meta).strip()))
        _CATALOG_CACHE[cache_key] = (len(corpus), catalog)
    for document_id, title in _CATALOG_CACHE[cache_key][1]:
        title_tokens = _tokens(title)
        title_all_tokens = _tokens(title, keep_generic=True)
        shared = query_tokens & title_tokens
        title_phrase = _canonical(title)
        # A subject is strong only when two meaningful terms overlap, or when a
        # multiword query phrase is literally present in the document title.
        phrase_match = len(query_tokens) >= 2 and query_phrase and query_phrase in title_phrase
        containment = len(query_title_tokens) >= 2 and query_title_tokens <= title_all_tokens
        paired_overlap = len(query_title_tokens & title_all_tokens) >= 2 and bool(query_tokens & title_all_tokens)
        if phrase_match or containment or paired_overlap or len(shared) >= 2:
            reason = "strong_title_phrase_overlap" if phrase_match else "strong_title_subject_overlap"
            return CatalogProbeResult(True, True, document_id, title or None, reason, round((time.perf_counter() - started) * 1000, 3))
    return CatalogProbeResult(True, False, reason="no_strong_catalog_match", timing_ms=round((time.perf_counter() - started) * 1000, 3))
