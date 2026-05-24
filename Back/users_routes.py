# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 17:57:25 2025

@author: aejau
"""

# users_routes.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import httpx

from auth_ms import require_access_token, verify_id_token

router = APIRouter()
_http = httpx.Client(timeout=10.0)

# --------- 1) Sync user (ID token) ---------
class SyncIn(BaseModel):
    id_token: str

@router.post("/users/sync")
def users_sync(body: SyncIn):
    """
    Reçoit l'ID token Microsoft depuis le front, le vérifie,
    et renvoie les infos de l'utilisateur (à persister si tu as une DB).
    """
    p = verify_id_token(body.id_token)
    user = {
        "sub": p.get("sub") or p.get("oid"),
        "email": p.get("preferred_username") or p.get("email"),
        "name": p.get("name"),
        "provider": "microsoft",
    }
    # TODO: upsert DB si besoin
    return {"ok": True, "user": user}

# --------- 2) Route protégée (exemple) ---------
@router.get("/me")
def me(user=Depends(require_access_token)):
    """
    Exemple de route PROTÉGÉE : nécessite Authorization: Bearer <access_token>.
    'user' contient le payload du token.
    """
    return {
        "sub": user.get("sub") or user.get("oid"),
        "email": user.get("preferred_username") or user.get("email"),
        "name": user.get("name"),
        "scp": user.get("scp"),
    }

# --------- 3) (Optionnel) Proxy simple vers Microsoft Graph ---------
GRAPH = "https://graph.microsoft.com/v1.0"

@router.get("/me/messages")
def list_messages(user=Depends(require_access_token)):
    """
    Proxy de test: renvoie les 10 derniers messages via Graph.
    Le front envoie son access token → on le réutilise ici.
    """
    access_token = user.get("access_token") or None  # non présent: on le reconstruit via Depends (voir ci-dessous)
    # Si 'access_token' n'est pas dans le payload (selon implément), on lit depuis l'en-tête via Depends:
    # Ici, on refait une dépendance légère pour extraire le token directement de l'appel:
    # (FastAPI ne remonte pas le 'credentials' initial ici, donc on redemande)
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    creds: HTTPAuthorizationCredentials = HTTPBearer(auto_error=True)(None)  # type: ignore

    token = access_token or creds.credentials  # fallback

    r = _http.get(f"{GRAPH}/me/messages?$top=10", headers={"Authorization": f"Bearer {token}"})
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()
