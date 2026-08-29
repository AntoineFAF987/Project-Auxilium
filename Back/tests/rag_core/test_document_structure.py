import json
import zipfile

import faiss
import numpy as np
import pytest

from rag_core.document_model import DOCUMENT_SCHEMA_VERSION
from rag_core.document_parsing import parse_document
from rag_core.indexer import RAGIndexer
from rag_core.retrieval import fuse_contiguous_passages
from rag_core.structure_chunking import chunk_document


def test_markdown_hierarchy_and_chunks_do_not_cross_sections(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text(
        "# Policy\n\nIntroduction.\n\n## Retention\n\nThirty days.\n\n"
        "## Deletion\n\nDeletion is automatic.",
        encoding="utf-8",
    )

    document = parse_document(str(path))
    chunks = chunk_document(document)

    assert [section.title for section in document.sections] == ["Policy", "Retention", "Deletion"]
    assert document.sections[1].parent_id == document.sections[0].section_id
    assert chunks[1].heading_path == ["Policy", "Retention"]
    assert all("Thirty days" not in chunk.text or "Deletion is automatic" not in chunk.text for chunk in chunks)
    assert [chunk.order for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].previous_chunk_uid is None
    assert chunks[-1].next_chunk_uid is None
    assert all(chunks[i].next_chunk_uid == chunks[i + 1].chunk_uid for i in range(len(chunks) - 1))
    assert all(chunks[i + 1].previous_chunk_uid == chunks[i].chunk_uid for i in range(len(chunks) - 1))
    assert document.blocks[0].next_block_id == document.blocks[1].block_id
    assert document.blocks[1].previous_block_id == document.blocks[0].block_id


def test_oversized_native_unit_uses_existing_size_and_overlap_policy(tmp_path):
    path = tmp_path / "long.md"
    body = " ".join(f"Sentence {i}." for i in range(250))
    path.write_text(f"# One section\n\n{body}", encoding="utf-8")

    chunks = chunk_document(parse_document(str(path)))

    assert len(chunks) > 2
    assert all(len(chunk.text) <= 800 for chunk in chunks)
    assert len({chunk.section_id for chunk in chunks}) == 1


def test_email_uses_sidecar_fields_and_keeps_attachment_links(tmp_path):
    flattened = tmp_path / "emails_flattened"
    cache = tmp_path / "emails_cache"
    flattened.mkdir()
    cache.mkdir()
    path = flattened / "message.txt"
    path.write_text(
        "Subject: Roadmap\nFrom: Alice <alice@example.test>\n"
        "To: Bob <bob@example.test>\nDate: 2026-08-01\n\nFirst paragraph.\n\nSecond paragraph.",
        encoding="utf-8",
    )
    (cache / "message.json").write_text(json.dumps({
        "id": "message-1",
        "conversationId": "thread-7",
        "subject": "Roadmap",
        "from_": {"name": "Alice", "address": "alice@example.test"},
        "to": [{"name": "Bob", "address": "bob@example.test"}],
        "receivedDateTime": "2026-08-01T12:00:00Z",
        "attachment_paths": [str(tmp_path / "attachment.pdf")],
    }), encoding="utf-8")

    document = parse_document(str(path))

    assert document.source == "email"
    assert document.source_metadata["message_id"] == "message-1"
    assert document.source_metadata["thread_id"] == "thread-7"
    assert document.source_metadata["attachment_paths"] == [str(tmp_path / "attachment.pdf")]
    assert document.source_metadata["attachments"][0]["relation_type"] == "attachment"
    assert document.source_metadata["attachments"][0]["document_id"].startswith("doc_")
    assert [block.block_type for block in document.blocks] == ["email_header", "email_body", "email_body"]
    assert all(block.source_metadata["message_order"] == 0 for block in document.blocks)


def test_docx_preserves_heading_paragraph_and_list(tmp_path):
    path = tmp_path / "structured.docx"
    document_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
      <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Overview</w:t></w:r></w:p>
      <w:p><w:r><w:t>Normal paragraph.</w:t></w:r></w:p>
      <w:p><w:pPr><w:numPr/></w:pPr><w:r><w:t>List entry</w:t></w:r></w:p>
      <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Cell A</w:t></w:r></w:p></w:tc>
      <w:tc><w:p><w:r><w:t>Cell B</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
    </w:body></w:document>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    document = parse_document(str(path))

    assert document.sections[0].title == "Overview"
    assert [block.block_type for block in document.blocks] == ["heading", "paragraph", "list_item", "table"]
    assert document.blocks[-1].text == "Cell A | Cell B"


def test_pdf_pages_are_native_chunk_boundaries(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "pages.pdf"
    pdf = fitz.open()
    for text in ("Page one paragraph.", "Page two paragraph."):
        page = pdf.new_page()
        page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()

    document = parse_document(str(path))
    chunks = chunk_document(document)

    assert {block.page for block in document.blocks} == {1, 2}
    assert len(chunks) == 2
    assert [(chunk.page, chunk.page_end) for chunk in chunks] == [(1, 1), (2, 2)]


def test_pdfplumber_fallback_preserves_page_numbers(tmp_path, monkeypatch):
    import pdfplumber

    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class Pdf:
        pages = [Page("First page."), Page("Second page.")]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    path = tmp_path / "fallback.pdf"
    path.write_bytes(b"not read because pdfplumber is mocked")
    monkeypatch.setattr(pdfplumber, "open", lambda _path: Pdf())

    document = parse_document(str(path))

    assert [(block.page, block.text) for block in document.blocks] == [
        (1, "First page."), (2, "Second page.")
    ]


def test_index_rows_keep_legacy_fields_and_add_structural_metadata(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Scope\n\nA coherent paragraph.", encoding="utf-8")
    indexer = object.__new__(RAGIndexer)

    rows, texts = indexer._rows_for_path(str(path))

    assert texts == [rows[0]["text"]]
    assert rows[0]["schema_version"] == DOCUMENT_SCHEMA_VERSION
    assert {"file", "path", "fingerprint", "chunk_id", "text_len", "source", "text"} <= rows[0].keys()
    assert rows[0]["document_id"]
    assert rows[0]["chunk_uid"]
    assert rows[0]["section_id"]
    assert "previous_chunk_uid" in rows[0]
    assert "next_chunk_uid" in rows[0]
    assert rows[0]["blocks"][0]["next_block_id"]


def test_schema_v1_incremental_update_forces_clean_rebuild(monkeypatch):
    indexer = object.__new__(RAGIndexer)
    indexer.loaded_schema_version = 1
    indexer.roots = ["C:/documents"]
    monkeypatch.setattr(indexer, "_load_existing", lambda: True)
    rebuilt = []
    monkeypatch.setattr(indexer, "build_or_update", lambda: rebuilt.append(True))

    indexer.rebuild_incremental()

    assert rebuilt == [True]


def test_manifest_versions_a_saved_index(tmp_path):
    indexer = object.__new__(RAGIndexer)
    indexer.corpus_path = str(tmp_path / "corpus.jsonl")
    indexer.emb_path = str(tmp_path / "embeddings.npy")
    indexer.faiss_path = str(tmp_path / "faiss.index")
    indexer.manifest_path = str(tmp_path / "manifest.json")
    indexer.index_dir = str(tmp_path)
    indexer.legacy_corpus_path = indexer.corpus_path
    indexer.legacy_emb_path = indexer.emb_path
    indexer.legacy_faiss_path = indexer.faiss_path
    indexer.loaded_schema_version = 1
    faiss_index = faiss.IndexFlatIP(2)

    indexer._save_all([], np.zeros((0, 2), dtype="float32"), faiss_index)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == DOCUMENT_SCHEMA_VERSION
    assert manifest["chunking"] == "structure-aware"
    assert manifest["chunk_max_chars"] == 800
    assert manifest["oversized_unit_overlap"] == 200
    assert manifest["chunk_count"] == 0
    assert manifest["generation"].startswith("v2-")
    assert (tmp_path / manifest["artifacts"]["corpus"]).exists()


def test_fusion_uses_explicit_neighbors_without_crossing_structural_zone():
    def meta(chunk_id, uid, previous, section, text):
        return {
            "path": "C:/docs/policy.md", "file": "policy.md",
            "document_id": "doc_1", "chunk_id": chunk_id, "order": chunk_id,
            "chunk_uid": uid, "previous_chunk_uid": previous,
            "section_id": section, "page": None, "text": text,
        }

    fused = fuse_contiguous_passages([
        (0.9, meta(0, "c0", None, "s1", "A")),
        (0.8, meta(1, "c1", "c0", "s1", "B")),
        (0.7, meta(2, "c2", "c1", "s2", "C")),
    ])

    assert [item[1]["text"] for item in fused] == ["A\n\nB", "C"]
