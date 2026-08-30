"""Safe local-file opening endpoints."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .source_references import local_source_by_document_id


router = APIRouter(prefix="/sources", tags=["sources"])
idx: Any | None = None


class SourceOpenRequest(BaseModel):
    document_id: str
    action: Literal["open_file", "reveal_in_folder"]


def _index() -> Any:
    """Delay model/index loading until this optional local desktop action is used."""
    global idx
    if idx is None:
        from .index_singleton import idx as loaded_index
        idx = loaded_index
    return idx


@router.post("/open")
def open_source(payload: SourceOpenRequest) -> dict[str, object]:
    # The client has no path field by design. Resolve only from the loaded index.
    source = local_source_by_document_id(getattr(_index(), "corpus", []), payload.document_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source locale indexée introuvable")
    origin_path = source.get("origin_path")
    if not origin_path:
        raise HTTPException(status_code=409, detail="Emplacement d'origine inconnu pour cette source")
    if not source.get("exists"):
        raise HTTPException(status_code=404, detail="Source introuvable à son emplacement d'origine")

    if payload.action == "open_file":
        startfile = getattr(os, "startfile", None)
        if startfile is None:
            raise HTTPException(status_code=501, detail="Ouverture de fichier disponible uniquement sous Windows")
        startfile(origin_path)
    else:
        subprocess.Popen(["explorer.exe", f'/select,"{origin_path}"'])
    return {"ok": True, "document_id": source["document_id"], "action": payload.action}
