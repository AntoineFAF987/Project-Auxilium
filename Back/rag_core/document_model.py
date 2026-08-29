from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


DOCUMENT_SCHEMA_VERSION = 2


@dataclass
class DocumentBlock:
    block_id: str
    document_id: str
    order: int
    text: str
    block_type: str = "paragraph"
    section_id: Optional[str] = None
    parent_id: Optional[str] = None
    previous_block_id: Optional[str] = None
    next_block_id: Optional[str] = None
    page: Optional[int] = None
    heading_path: List[str] = field(default_factory=list)
    source_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentSection:
    section_id: str
    document_id: str
    order: int
    title: Optional[str] = None
    level: Optional[int] = None
    parent_id: Optional[str] = None
    previous_section_id: Optional[str] = None
    next_section_id: Optional[str] = None
    heading_path: List[str] = field(default_factory=list)
    block_ids: List[str] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class Document:
    document_id: str
    source: str
    path: str
    file: str
    title: Optional[str] = None
    sections: List[DocumentSection] = field(default_factory=list)
    blocks: List[DocumentBlock] = field(default_factory=list)
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    parent_document_id: Optional[str] = None
    relation_type: Optional[str] = None


@dataclass
class Chunk:
    chunk_uid: str
    document_id: str
    chunk_id: int
    order: int
    text: str
    block_ids: List[str]
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    section_id: Optional[str] = None
    section: Optional[str] = None
    heading_path: List[str] = field(default_factory=list)
    page: Optional[int] = None
    page_end: Optional[int] = None
    parent_id: Optional[str] = None
    previous_chunk_uid: Optional[str] = None
    next_chunk_uid: Optional[str] = None
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
