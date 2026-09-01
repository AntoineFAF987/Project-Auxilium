import os
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

_BACK_ROOT = Path(__file__).resolve().parents[2]
if "api" not in sys.modules:
    package = types.ModuleType("api")
    package.__path__ = [str(_BACK_ROOT / "api")]
    sys.modules["api"] = package

from api.source_references import local_source_by_document_id, references_for_blocks, select_cited_references
from api import routes_sources
from api.routes_sources import SourceOpenRequest, open_source


def _local_row(path="C:/Users/Alice/Documents/SAMSON/manuel.pdf", document_id="doc_manual"):
    return {
        "document_id": document_id,
        "source": "pdf",
        "file": "manuel.pdf",
        "path": path,
        "document_metadata": {"origin_path": path, "indexed_path": "C:/Auxilium/Back/data/documents_ingested/manuel.pdf"},
    }


def test_local_reference_has_user_origin_contract(tmp_path):
    origin = tmp_path / "SAMSON" / "manuel.pdf"
    origin.parent.mkdir()
    origin.write_text("ok")
    ref = local_source_by_document_id([_local_row(str(origin))], "doc_manual")
    assert ref == {
        "document_id": "doc_manual", "type": "local_file", "display_name": "manuel.pdf",
        "origin_path": str(origin.resolve()), "folder_path": str(origin.parent.resolve()),
        "indexed_path": "C:/Auxilium/Back/data/documents_ingested/manuel.pdf", "exists": True,
    }


def test_multiple_chunks_are_deduplicated_by_document_id():
    rows = [_local_row(), {**_local_row(), "chunk_id": 1}]
    assert len(references_for_blocks([(1.0, row) for row in rows])) == 1


def test_eight_chunks_map_to_two_final_documents_in_citation_order():
    doc_a = _local_row("C:/Docs/A.pdf", "doc_a")
    doc_b = _local_row("C:/Docs/B.pdf", "doc_b")
    blocks = [(1.0, {**(doc_a if index in {1, 3, 6} else doc_b), "chunk_id": index}) for index in range(1, 9)]
    sources, mapping = select_cited_references(blocks, [1, 3, 6, 8])
    assert [source["document_id"] for source in sources] == ["doc_a", "doc_b"]
    assert mapping == {1: 1, 2: 2, 3: 1, 4: 2, 5: 2, 6: 1, 7: 2, 8: 2}


def test_unselected_document_has_no_public_citation_mapping():
    doc_a = _local_row("C:/Docs/A.pdf", "doc_a")
    doc_b = _local_row("C:/Docs/B.pdf", "doc_b")
    blocks = [(1.0, {**(doc_a if index < 3 else doc_b), "chunk_id": index}) for index in range(1, 5)]
    sources, mapping = select_cited_references(blocks, [1])
    assert [source["document_id"] for source in sources] == ["doc_a"]
    assert mapping == {1: 1, 2: 1}


def test_legacy_direct_file_path_is_migrated_without_reindexing(tmp_path):
    origin = tmp_path / "legacy.pdf"
    origin.write_text("ok")
    ref = references_for_blocks([(1.0, {"path": str(origin), "file": "legacy.pdf", "chunk_id": 0})])[0]
    assert ref["type"] == "local_file"
    assert ref["origin_path"] == str(origin.resolve())
    assert ref["exists"] is True


def test_open_and_reveal_resolve_only_document_id(tmp_path, monkeypatch):
    origin = tmp_path / "manuel.pdf"
    origin.write_text("ok")
    monkeypatch.setattr(routes_sources, "idx", types.SimpleNamespace(corpus=[_local_row(str(origin))]))
    with patch.object(os, "startfile", create=True) as startfile:
        assert open_source(SourceOpenRequest(document_id="doc_manual", action="open_file"))["ok"] is True
        startfile.assert_called_once_with(str(origin.resolve()))
    with patch("api.routes_sources.subprocess.Popen") as popen:
        assert open_source(SourceOpenRequest(document_id="doc_manual", action="reveal_in_folder"))["ok"] is True
        popen.assert_called_once()


@pytest.mark.parametrize("corpus, document_id, detail", [
    ([], "unknown", "introuvable"),
    ([{"document_id": "email", "source": "email", "path": "x"}], "email", "introuvable"),
    ([{"document_id": "missing", "source": "pdf", "path": "C:/Auxilium/Back/data/documents_ingested/x.pdf"}], "missing", "Emplacement d'origine inconnu"),
])
def test_open_rejects_unknown_nonlocal_or_missing_origin(monkeypatch, corpus, document_id, detail):
    monkeypatch.setattr(routes_sources, "idx", types.SimpleNamespace(corpus=corpus))
    with pytest.raises(HTTPException, match=detail):
        open_source(SourceOpenRequest(document_id=document_id, action="open_file"))


def test_deleted_origin_is_reported_and_never_opened(tmp_path, monkeypatch):
    origin = tmp_path / "deleted.pdf"
    row = _local_row(str(origin))
    assert local_source_by_document_id([row], "doc_manual")["exists"] is False
    monkeypatch.setattr(routes_sources, "idx", types.SimpleNamespace(corpus=[row]))
    with pytest.raises(HTTPException, match="Source introuvable"):
        open_source(SourceOpenRequest(document_id="doc_manual", action="open_file"))


def test_request_model_has_no_client_controlled_path_field():
    assert "path" not in SourceOpenRequest.model_fields
