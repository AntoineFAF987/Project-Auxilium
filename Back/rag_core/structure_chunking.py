from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import List, Optional, Tuple

from ingest_pdfs import CHUNK_CHARS, CHUNK_OVERLAP, split_text_fast

from .document_model import Chunk, Document, DocumentBlock


def _boundary(block: DocumentBlock, source: str) -> Tuple[Optional[str], Optional[int], str]:
    # Page is a native boundary for PDFs; section/message is the boundary elsewhere.
    page = block.page if source == "pdf" else None
    message = str(block.source_metadata.get("message_order", "")) if source == "email" else ""
    return block.section_id, page, message


def _uid(document_id: str, first_block_id: str, part: int) -> str:
    digest = hashlib.sha256(f"{document_id}|{first_block_id}|{part}".encode("utf-8")).hexdigest()[:16]
    return f"{document_id}:c{digest}"


def chunk_document(document: Document, max_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """Group native blocks first; split only oversized semantic units.

    The existing 800/200 limits remain the size policy. Overlap is applied only
    when one native block itself must be split, never across section/page/message
    boundaries.
    """
    groups: List[List[DocumentBlock]] = []
    current: List[DocumentBlock] = []
    current_len = 0
    current_boundary = None
    for block in document.blocks:
        boundary = _boundary(block, document.source)
        addition = len(block.text) + (2 if current else 0)
        if current and (boundary != current_boundary or current_len + addition > max_chars):
            groups.append(current)
            current, current_len = [], 0
        current_boundary = boundary
        current.append(block)
        current_len += len(block.text) + (2 if len(current) > 1 else 0)
    if current:
        groups.append(current)

    chunks: List[Chunk] = []
    for group in groups:
        text = "\n\n".join(block.text for block in group).strip()
        pieces = split_text_fast(text, max_chars=max_chars, overlap=overlap) if len(text) > max_chars else [text]
        for part, piece in enumerate(pieces):
            first, last = group[0], group[-1]
            section_title = first.heading_path[-1] if first.heading_path else None
            chunks.append(Chunk(
                chunk_uid=_uid(document.document_id, first.block_id, part),
                document_id=document.document_id, chunk_id=len(chunks), order=len(chunks),
                text=piece, block_ids=[block.block_id for block in group],
                blocks=[asdict(block) for block in group],
                section_id=first.section_id, section=section_title,
                heading_path=list(first.heading_path), page=first.page, page_end=last.page,
                parent_id=first.section_id or document.document_id,
                source_metadata={
                    "block_start_order": first.order,
                    "block_end_order": last.order,
                    "block_types": list(dict.fromkeys(block.block_type for block in group)),
                },
            ))
    for index, chunk in enumerate(chunks):
        chunk.previous_chunk_uid = chunks[index - 1].chunk_uid if index else None
        chunk.next_chunk_uid = chunks[index + 1].chunk_uid if index + 1 < len(chunks) else None
    return chunks
