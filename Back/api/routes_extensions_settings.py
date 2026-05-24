# -*- coding: utf-8 -*-
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from .extensions_db import get_allowed_extensions, set_allowed_extensions

router = APIRouter()

# ---- Schémas ----
class ExtsOut(BaseModel):
    extensions: List[str]

class ExtsIn(BaseModel):
    extensions: List[str]
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None

# ---- Endpoints ----
@router.get("/settings/extensions", response_model=ExtsOut)
def get_exts(tenant_id: Optional[str] = Query(None), user_id: Optional[str] = Query(None)):
    exts = get_allowed_extensions(tenant_id=tenant_id, user_id=user_id)
    return {"extensions": exts}

@router.put("/settings/extensions", response_model=ExtsOut)
def put_exts(body: ExtsIn):
    try:
        saved = set_allowed_extensions(body.extensions or [], tenant_id=body.tenant_id, user_id=body.user_id)
        return {"extensions": saved}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible d'enregistrer: {e}")
