"""Source references exposed by the API (never paths supplied by a client)."""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Any, Iterable, TypedDict


LOCAL_SOURCES = {"file", "pdf"}


class SourceReference(TypedDict, total=False):
    """Extensible API contract: local_file now; email/web later."""

    document_id: str
    type: str
    display_name: str
    origin_path: str | None
    folder_path: str | None
    indexed_path: str | None
    exists: bool


def _is_internal_ingestion_path(path: str) -> bool:
    """Old copied documents under Back/data are not user-origin locations."""
    normalized = path.replace("\\", "/").casefold()
    return "/auxilium/back/data/" in normalized


def _legacy_local_path(path: object) -> bool:
    normalized = str(path or "").replace("\\", "/").casefold()
    return (
        Path(normalized).suffix in {".pdf", ".docx", ".txt", ".md"}
        and not any(part in normalized for part in ("/emails_flattened/", "/emails_cache/", "/email_attachments/"))
    )


def _document_id(metadata: dict[str, Any]) -> str:
    existing = str(metadata.get("document_id") or "")
    if existing:
        return existing
    path = str(metadata.get("path") or "")
    # Lightweight compatibility identifier for v1 direct-file rows. It is
    # deterministic and only resolves back through the already loaded corpus.
    return "legacy_" + hashlib.sha256(path.casefold().encode("utf-8")).hexdigest()[:24] if path else ""


def source_reference(metadata: dict[str, Any]) -> SourceReference:
    """Build the public, structured reference for one indexed document."""
    source = str(metadata.get("source") or "")
    if not source and _legacy_local_path(metadata.get("path")):
        source = "file"
    if source not in LOCAL_SOURCES:
        return {
            "document_id": _document_id(metadata),
            "type": source or "unknown",
            "display_name": metadata.get("title") or metadata.get("file") or "Source",
        }

    document_metadata = metadata.get("document_metadata")
    document_metadata = document_metadata if isinstance(document_metadata, dict) else {}
    indexed_path = document_metadata.get("indexed_path") or metadata.get("path")
    origin_path = document_metadata.get("origin_path")
    # Legacy direct indexing used path for the original selected file.  Do not
    # mistake known Auxilium ingestion copies for an origin path.
    if not origin_path and indexed_path and not _is_internal_ingestion_path(str(indexed_path)):
        origin_path = str(indexed_path)
    origin_path = os.path.abspath(str(origin_path)) if origin_path else None
    return {
        "document_id": _document_id(metadata),
        "type": "local_file",
        "display_name": str(document_metadata.get("display_name") or metadata.get("file") or metadata.get("title") or Path(origin_path or indexed_path or "Source").name),
        "origin_path": origin_path,
        "folder_path": str(Path(origin_path).parent) if origin_path else None,
        "indexed_path": str(indexed_path) if indexed_path else None,
        "exists": bool(origin_path and Path(origin_path).is_file()),
    }


def local_source_by_document_id(corpus: Iterable[dict[str, Any]], document_id: str) -> SourceReference | None:
    """Resolve a local source from indexed metadata; document_id is the only key."""
    for row in corpus:
        if _document_id(row) == str(document_id):
            reference = source_reference(row)
            return reference if reference.get("type") == "local_file" else None
    return None


def references_for_blocks(blocks: Iterable[tuple[float, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Return one source per document, preserving citation/retrieval order.

    Non-local sources deliberately retain the legacy shape: email handling is
    out of scope for this change and therefore remains behavior-compatible.
    """
    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, metadata in blocks:
        source = str(metadata.get("source") or "")
        if not source and _legacy_local_path(metadata.get("path")):
            source = "file"
        if source not in LOCAL_SOURCES:
            path = metadata.get("path")
            if path and str(path) not in seen:
                seen.add(str(path))
                references.append({"path": path, "chunk": -1})
            continue
        reference = source_reference(metadata)
        document_id = str(reference.get("document_id") or "")
        key = document_id or str(reference.get("indexed_path") or reference.get("origin_path") or "")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        references.append(reference)
    return references


def select_cited_references(
    blocks: list[tuple[float, dict[str, Any]]], cited_chunk_indices: Iterable[int],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Expose documents in citation order and map RAG chunk ids to UI ids.

    Context indices are chunk-based because that is what the LLM sees. The UI
    exposes one item per document, so this is where raw chunk citations become
    stable, user-visible source numbers.
    """
    requested = [int(index) for index in cited_chunk_indices if 1 <= int(index) <= len(blocks)]
    candidate_indices = list(dict.fromkeys(requested)) or list(range(1, len(blocks) + 1))

    def reference_for(index: int) -> dict[str, Any] | None:
        refs = references_for_blocks([blocks[index - 1]])
        return refs[0] if refs else None

    def reference_key(reference: dict[str, Any]) -> str:
        return str(reference.get("document_id") or reference.get("path") or reference.get("indexed_path") or reference.get("origin_path") or "")

    references: list[dict[str, Any]] = []
    display_by_key: dict[str, int] = {}
    for index in candidate_indices:
        reference = reference_for(index)
        if reference is None:
            continue
        key = reference_key(reference)
        if key and key not in display_by_key:
            display_by_key[key] = len(references) + 1
            references.append(reference)

    chunk_to_display: dict[int, int] = {}
    for index in range(1, len(blocks) + 1):
        reference = reference_for(index)
        if reference is None:
            continue
        visible = display_by_key.get(reference_key(reference))
        if visible is not None:
            chunk_to_display[index] = visible
    return references, chunk_to_display
