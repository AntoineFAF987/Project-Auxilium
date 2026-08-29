from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

from .document_model import Document, DocumentBlock, DocumentSection
from .utils import read_supported_text


_MARKDOWN_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+] |\d+[.)]\s+)")
_EMAIL_HEADER = re.compile(r"^(Subject|From|To|CC|Date|Categories):\s*(.*)$", re.I)


def stable_document_id(path: str, native_id: Optional[str] = None) -> str:
    identity = native_id or str(Path(path).resolve()).casefold()
    return "doc_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


class _Builder:
    def __init__(self, path: str, source: str, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, native_id: Optional[str] = None):
        self.document = Document(
            document_id=stable_document_id(path, native_id), source=source,
            path=str(Path(path).resolve()), file=Path(path).name, title=title,
            source_metadata=metadata or {},
        )
        self._heading_stack: List[Tuple[int, str, str]] = []
        self._section_order = 0

    def heading(self, text: str, level: int, page: Optional[int] = None) -> None:
        text = text.strip()
        while self._heading_stack and self._heading_stack[-1][0] >= level:
            self._heading_stack.pop()
        section_id = f"{self.document.document_id}:s{self._section_order:05d}"
        parent_id = self._heading_stack[-1][2] if self._heading_stack else None
        path = [item[1] for item in self._heading_stack] + [text]
        self.document.sections.append(DocumentSection(
            section_id=section_id, document_id=self.document.document_id,
            order=self._section_order, title=text, level=level,
            parent_id=parent_id, heading_path=path, page_start=page, page_end=page,
        ))
        if len(self.document.sections) > 1:
            previous = self.document.sections[-2]
            previous.next_section_id = section_id
            self.document.sections[-1].previous_section_id = previous.section_id
        self._section_order += 1
        self._heading_stack.append((level, text, section_id))
        self.block(text, "heading", page=page, source_metadata={"heading_level": level})

    def block(self, text: str, block_type: str = "paragraph", *, page: Optional[int] = None, source_metadata: Optional[Dict[str, Any]] = None) -> None:
        text = (text or "").strip()
        if not text:
            return
        order = len(self.document.blocks)
        current = self._heading_stack[-1] if self._heading_stack else None
        section_id = current[2] if current else None
        heading_path = [item[1] for item in self._heading_stack]
        block_id = f"{self.document.document_id}:b{order:06d}"
        block = DocumentBlock(
            block_id=block_id, document_id=self.document.document_id, order=order,
            text=text, block_type=block_type, section_id=section_id,
            parent_id=section_id, page=page, heading_path=heading_path,
            source_metadata=source_metadata or {},
        )
        if self.document.blocks:
            previous = self.document.blocks[-1]
            previous.next_block_id = block_id
            block.previous_block_id = previous.block_id
        self.document.blocks.append(block)
        if section_id:
            section = next(s for s in reversed(self.document.sections) if s.section_id == section_id)
            section.block_ids.append(block_id)
            if page is not None:
                section.page_start = page if section.page_start is None else min(section.page_start, page)
                section.page_end = page if section.page_end is None else max(section.page_end, page)


def _paragraphs(text: str) -> Iterable[str]:
    for value in re.split(r"\n\s*\n", text or ""):
        value = value.strip()
        if value:
            yield value


def _email_sidecar(path: Path) -> Dict[str, Any]:
    candidates = [path.with_suffix(".json")]
    if path.parent.name == "emails_flattened":
        candidates.append(path.parent.parent / "emails_cache" / (path.stem + ".json"))
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and (payload.get("id") or payload.get("subject")):
                return payload
        except Exception:
            continue
    return {}


def _parse_email(path: str, raw: str) -> Document:
    sidecar = _email_sidecar(Path(path))
    lines = raw.splitlines()
    headers: Dict[str, str] = {}
    body_at = 0
    for i, line in enumerate(lines):
        if not line.strip():
            body_at = i + 1
            break
        match = _EMAIL_HEADER.match(line)
        if not match:
            break
        headers[match.group(1).lower()] = match.group(2).strip()
        body_at = i + 1
    title = sidecar.get("subject") or headers.get("subject") or Path(path).stem
    native_id = sidecar.get("id")
    attachment_paths = sidecar.get("attachment_paths") or []
    metadata = {
        "message_id": native_id,
        "thread_id": sidecar.get("conversationId"),
        "subject": title,
        "sender": sidecar.get("from_") or headers.get("from"),
        "recipients": sidecar.get("to") or headers.get("to"),
        "cc": sidecar.get("cc") or headers.get("cc"),
        "date": sidecar.get("receivedDateTime") or headers.get("date"),
        "chronological_key": sidecar.get("receivedDateTime") or headers.get("date"),
        "folder": sidecar.get("folder"),
        "user": sidecar.get("user"),
        "attachment_paths": attachment_paths,
        "attachments": [
            {
                "path": str(Path(attachment_path).resolve()),
                "document_id": stable_document_id(attachment_path),
                "relation_type": "attachment",
            }
            for attachment_path in attachment_paths
        ],
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "", [])}
    builder = _Builder(path, "email", title, metadata, native_id=native_id)
    header_text = "\n".join(lines[:body_at]).strip()
    builder.block(header_text, "email_header", source_metadata={"message_order": 0})
    body = "\n".join(lines[body_at:]).strip()
    for paragraph in _paragraphs(body):
        builder.block(paragraph, "email_body", source_metadata={"message_order": 0})
    return builder.document


def _parse_docx(path: str) -> Document:
    builder = _Builder(path, "file", Path(path).stem)
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    body = next(node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "body")
    # Walk top-level Word flow only; paragraphs inside a table are represented
    # by the table block and must not be duplicated as standalone paragraphs.
    for element in list(body):
        local = element.tag.rsplit("}", 1)[-1]
        if local == "p":
            text = "".join(node.text or "" for node in element.iter() if node.tag.rsplit("}", 1)[-1] == "t").strip()
            if not text:
                continue
            style = next((node.attrib.get(next((k for k in node.attrib if k.endswith("}val")), ""), "") for node in element.iter() if node.tag.rsplit("}", 1)[-1] == "pStyle"), "")
            match = re.search(r"(?:Heading|Titre)\s*([1-6])", style, re.I)
            if match:
                builder.heading(text, int(match.group(1)))
            else:
                is_list = any(node.tag.rsplit("}", 1)[-1] == "numPr" for node in element.iter())
                builder.block(text, "list_item" if is_list else "paragraph", source_metadata={"style": style} if style else {})
        elif local == "tbl":
            rows = []
            for row in [n for n in element if n.tag.rsplit("}", 1)[-1] == "tr"]:
                cells = []
                for cell in [n for n in row if n.tag.rsplit("}", 1)[-1] == "tc"]:
                    cells.append(" ".join(t.text or "" for t in cell.iter() if t.tag.rsplit("}", 1)[-1] == "t").strip())
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                builder.block("\n".join(rows), "table")
    return builder.document


def _parse_pdf(path: str) -> Document:
    builder = _Builder(path, "pdf", Path(path).stem)
    from ingest_pdfs import HARD_LIMIT_PAGES, MAX_TEXT_CHARS_PER_PDF
    extracted_chars = 0

    def remaining(text: str) -> str:
        nonlocal extracted_chars
        if MAX_TEXT_CHARS_PER_PDF > 0:
            text = text[:max(0, MAX_TEXT_CHARS_PER_PDF - extracted_chars)]
        extracted_chars += len(text)
        return text

    try:
        import fitz
        with fitz.open(path) as pdf:
            limit = len(pdf) if HARD_LIMIT_PAGES <= 0 else min(len(pdf), HARD_LIMIT_PAGES)
            for page_index in range(limit):
                if MAX_TEXT_CHARS_PER_PDF > 0 and extracted_chars >= MAX_TEXT_CHARS_PER_PDF:
                    break
                page = pdf[page_index]
                data = page.get_text("dict")
                candidates = []
                for raw_block in data.get("blocks", []):
                    if raw_block.get("type") != 0:
                        continue
                    lines, sizes = [], []
                    for line in raw_block.get("lines", []):
                        spans = line.get("spans", [])
                        line_text = "".join(span.get("text", "") for span in spans).strip()
                        if line_text:
                            lines.append(line_text)
                            sizes.extend(float(span.get("size", 0)) for span in spans if span.get("text", "").strip())
                    text = "\n".join(lines).strip()
                    if text:
                        candidates.append((text, max(sizes or [0.0]), raw_block.get("bbox")))
                body_sizes = sorted(size for text, size, _ in candidates if len(text) > 80 and size > 0)
                baseline = body_sizes[len(body_sizes) // 2] if body_sizes else 0.0
                for text, size, bbox in candidates:
                    text = remaining(text)
                    if not text:
                        break
                    is_heading = bool(baseline and size >= baseline * 1.18 and len(text) <= 160 and text.count("\n") <= 2)
                    meta = {"bbox": list(bbox) if bbox else None, "font_size": size}
                    if is_heading:
                        builder.heading(text.replace("\n", " "), 1, page=page_index + 1)
                    else:
                        builder.block(text, "paragraph", page=page_index + 1, source_metadata=meta)
        return builder.document
    except Exception:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            limit = len(pdf.pages) if HARD_LIMIT_PAGES <= 0 else min(len(pdf.pages), HARD_LIMIT_PAGES)
            for page_index in range(limit):
                if MAX_TEXT_CHARS_PER_PDF > 0 and extracted_chars >= MAX_TEXT_CHARS_PER_PDF:
                    break
                raw = pdf.pages[page_index].extract_text() or ""
                for paragraph in _paragraphs(raw):
                    paragraph = remaining(paragraph)
                    if not paragraph:
                        break
                    builder.block(paragraph, page=page_index + 1)
        return builder.document


def _parse_text(path: str, raw: str) -> Document:
    builder = _Builder(path, "file", Path(path).stem)
    pending: List[str] = []

    def flush() -> None:
        if pending:
            text = "\n".join(pending).strip()
            builder.block(text, "list" if all(_LIST_ITEM.match(line) for line in pending if line.strip()) else "paragraph")
            pending.clear()

    for line in raw.splitlines():
        heading = _MARKDOWN_HEADING.match(line)
        if heading:
            flush()
            builder.heading(heading.group(2), len(heading.group(1)))
        elif not line.strip():
            flush()
        else:
            pending.append(line)
    flush()
    return builder.document


def parse_document(path: str) -> Document:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(path)
    if ext == ".docx":
        return _parse_docx(path)
    raw = read_supported_text(path)
    if ext in {".txt", ".md"} and raw.lstrip().lower().startswith("subject:"):
        return _parse_email(path, raw)
    return _parse_text(path, raw)
