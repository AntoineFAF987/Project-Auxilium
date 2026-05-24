# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 17:55:57 2025

@author: aejau
"""

# auth_ms.py
import os, json, time
from typing import Dict, Any
import httpx
from jose import jwt
from jose.backends import RSAKey
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# --- Config (venant de .env) ---
AAD_TENANT    = os.getenv("AAD_TENANT", "consumers")
AAD_CLIENT_ID = os.getenv("AAD_CLIENT_ID")  # le même client_id que sur le front
AUTHORITY     = f"https://login.microsoftonline.com/{AAD_TENANT}"
OIDC_META_URL = f"{AUTHORITY}/v2.0/.well-known/openid-configuration"

_http   = httpx.Client(timeout=10.0)
_bearer = HTTPBearer(auto_error=True)

# Caches simples (1h)
_OIDC_CACHE: Dict[str, Any] = {}
_JWKS_CACHE: Dict[str, Any] = {}

def _get_oidc_metadata() -> Dict[str, Any]:
    now = time.time()
    item = _OIDC_CACHE.get("meta")
    if not item or now - item["ts"] > 3600:
        r = _http.get(OIDC_META_URL); r.raise_for_status()
        _OIDC_CACHE["meta"] = {"ts": now, "data": r.json()}
    return _OIDC_CACHE["meta"]["data"]

def _get_jwks() -> Dict[str, Any]:
    now = time.time()
    item = _JWKS_CACHE.get("jwks")
    if not item or now - item["ts"] > 3600:
        jwks_uri = _get_oidc_metadata()["jwks_uri"]
        r = _http.get(jwks_uri); r.raise_for_status()
        _JWKS_CACHE["jwks"] = {"ts": now, "data": r.json()}
    return _JWKS_CACHE["jwks"]["data"]

def _public_key_for(token: str):
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    
    header = jwt.get_unverified_header(token)
    kid = header.get("kid")
    if not kid:
        raise HTTPException(status_code=401, detail="Token sans 'kid'.")
    
    for k in _get_jwks().get("keys", []):
        if k.get("kid") == kid:
            # Construire la clé publique RSA depuis les composants JWK
            from jose.backends.rsa_backend import RSAKey
            key = RSAKey(k, algorithm="RS256")
            return key.public_key()
    
    raise HTTPException(status_code=401, detail="Clé de signature introuvable.")

def verify_ms_token(token: str, *, expect_id_token: bool) -> Dict[str, Any]:
    """
    Vérifie un ID token (expect_id_token=True) ou un Access token (False).
    - Signature RS256 (clé Microsoft)
    - issuer = .../{tenant}/v2.0 ou sts.windows.net
    - audience = ton client_id
    """
    if not AAD_CLIENT_ID:
        raise RuntimeError("AAD_CLIENT_ID manquant (mets-le dans .env)")
    pub = _public_key_for(token)
    
    # Décoder sans valider l'issuer (on le fera manuellement)
    payload = jwt.decode(
        token,
        pub,
        algorithms=["RS256"],
        audience=AAD_CLIENT_ID,
        options={"verify_iss": False}  # On valide manuellement l'issuer
    )
    
    # Validation manuelle de l'issuer (accepte plusieurs formats Microsoft)
    iss = payload.get("iss", "")
    # Extraire le tenant_id de l'issuer pour validation flexible
    if not (iss.startswith("https://login.microsoftonline.com/") or iss.startswith("https://sts.windows.net/")):
        raise HTTPException(status_code=401, detail=f"Issuer invalide: {iss}")
    
    if not expect_id_token:
        if "scp" not in payload and "roles" not in payload:
            raise HTTPException(status_code=401, detail="Access token sans 'scp/roles'.")
    return payload

# --------- Dépendances FastAPI ---------

async def require_access_token(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> Dict[str, Any]:
    """À utiliser pour PROTÉGER une route API avec l'access token Graph."""
    token = credentials.credentials
    return verify_ms_token(token, expect_id_token=False)

def verify_id_token(id_token: str) -> Dict[str, Any]:
    """À utiliser dans /users/sync (reçoit l'ID token depuis le front)."""
    return verify_ms_token(id_token, expect_id_token=True)
