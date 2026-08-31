# -*- coding: utf-8 -*-
from typing import Optional, List, Dict, Tuple
from fastapi import APIRouter, HTTPException, Path, Body, Request

from auth_ms import verify_ms_token
from .chats_db import create_chat, list_chats, list_messages, append_message, rename_chat, set_chat_project, set_chat_pinned, soft_delete_chat

router = APIRouter()

# ---- Helpers ----

def _try_get_auth_ids(request: Request) -> Tuple[str, str]:
    """Extrait (tenant_id, user_id) depuis Authorization ou lève une exception."""
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if not auth or " " not in auth:
        raise HTTPException(status_code=401, detail="Authorization header manquant")
    scheme, token = auth.split(" ", 1)
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Token invalide")
    try:
        # Essayer d'abord comme ID token, puis comme Access token
        try:
            p = verify_ms_token(token, expect_id_token=True)
        except:
            p = verify_ms_token(token, expect_id_token=False)
        tenant_id = p.get("tid") or p.get("tenant_id") or "default"
        user_id = p.get("oid") or p.get("sub") or "user"
        return str(tenant_id), str(user_id)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token invalide: {e}")

# ---- Routes ----

@router.get("/chats")
def api_list_chats(request: Request):
    tenant_id, user_id = _try_get_auth_ids(request)
    return {"items": list_chats(tenant_id, user_id)}

@router.post("/chats")
def api_create_chat(body: Dict = Body(default_factory=dict), request: Request = None):
    tenant_id, user_id = _try_get_auth_ids(request)
    title = (body.get("title") or "").strip() or None
    # Optionnel: permettre au client de fournir un id (pour concilier avec /ask.thread_id)
    provided_id = (body.get("id") or body.get("chat_id") or "").strip() or None
    project_id = body.get("project_id")
    if project_id is not None and (not isinstance(project_id, str) or not project_id.strip()):
        raise HTTPException(status_code=400, detail="project_id invalide")
    try:
        chat_id = create_chat(tenant_id, user_id, title, chat_id=provided_id, project_id=project_id.strip() if isinstance(project_id, str) else None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"id": chat_id, "title": title or "Nouveau chat", "project_id": project_id.strip() if isinstance(project_id, str) else None}

@router.get("/chats/{chat_id}/messages")
def api_list_messages(chat_id: str = Path(..., min_length=1), request: Request = None):
    tenant_id, user_id = _try_get_auth_ids(request)
    try:
        items = list_messages(tenant_id, user_id, chat_id)
        return {"items": items}
    except PermissionError:
        raise HTTPException(status_code=404, detail="Chat introuvable")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement des messages: {str(e)}")

@router.post("/chats/{chat_id}/messages")
def api_append_message(
    chat_id: str = Path(..., min_length=1),
    body: Dict = Body(...),
    request: Request = None
):
    tenant_id, user_id = _try_get_auth_ids(request)
    role = (body.get("role") or "").strip()
    content = body.get("content") or ""
    if role not in {"user", "assistant", "system"}:
        raise HTTPException(status_code=400, detail="role invalide")
    try:
        mid = append_message(tenant_id, user_id, chat_id, role, content, meta=body.get("meta"))
    except PermissionError:
        raise HTTPException(status_code=404, detail="Chat introuvable")
    return {"id": mid}

@router.patch("/chats/{chat_id}")
def api_update_chat(chat_id: str, body: Dict = Body(...), request: Request = None):
    tenant_id, user_id = _try_get_auth_ids(request)
    if "title" not in body and "project_id" not in body and "pinned" not in body:
        raise HTTPException(status_code=400, detail="Aucune modification demandee")
    ok = True
    if "title" in body:
        title = (body.get("title") or "").strip()
        if not title:
            raise HTTPException(status_code=400, detail="Titre vide")
        ok = rename_chat(tenant_id, user_id, chat_id, title)
    if ok and "project_id" in body:
        project_id = body.get("project_id")
        if project_id is not None and (not isinstance(project_id, str) or not project_id.strip()):
            raise HTTPException(status_code=400, detail="project_id invalide")
        try:
            ok = set_chat_project(tenant_id, user_id, chat_id, project_id.strip() if isinstance(project_id, str) else None)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    if ok and "pinned" in body:
        pinned = body.get("pinned")
        if not isinstance(pinned, bool):
            raise HTTPException(status_code=400, detail="pinned invalide")
        ok = set_chat_pinned(tenant_id, user_id, chat_id, pinned)
    if not ok:
        raise HTTPException(status_code=404, detail="Chat introuvable")
    return {"ok": True}

@router.delete("/chats/{chat_id}")
def api_delete_chat(chat_id: str, request: Request = None):
    tenant_id, user_id = _try_get_auth_ids(request)
    ok = soft_delete_chat(tenant_id, user_id, chat_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Chat introuvable")
    return {"ok": True}
