# -*- coding: utf-8 -*-
"""
API locale On-Prem pour piloter le moteur RAG.
"""
import os
from pathlib import Path

# Charger le .env depuis le dossier Back/
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid

from init_auth import wire_auth

# Routers existants
from .routes_status import router as status_router
from .routes_ingest import router as ingest_router
from .routes_ask import router as ask_router
from .routes_config import router as config_router
from .routes_email_settings import router as email_settings_router
from .routes_directory_settings import router as directory_settings_router
# ✅ NOUVEAU : Extensions
from .routes_extensions_settings import router as extensions_settings_router
from .routes_chats import router as chats_router
from .routes_projects import router as projects_router
from .routes_mail_sync import router as mail_sync_router
from .routes_orchestrator_debug import router as orchestrator_debug_router
from .routes_response_trace import router as response_trace_router
from .routes_sources import router as sources_router

# Rate limit & config
from .sessions import _RateLimiter
from .config import rate_limit

from .db import init_db
from .auto_sync_scheduler import start_auto_sync_scheduler, stop_auto_sync_scheduler

app = FastAPI(title="RAG On-Prem API", version="0.9")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

wire_auth(app)

@app.middleware("http")
async def request_id_mw(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response

MAX_REQ = int(rate_limit["max_req"])
PER_SEC = int(rate_limit["per_sec"])

# Initialiser le rate limiter avec les valeurs de config
RATE_LIMITER = _RateLimiter(max_req=MAX_REQ, per_sec=PER_SEC)
RATE_LIMIT_EXEMPT_GET_PATHS = {
    "/config",
    "/emails/available",
    "/settings/directories",
    "/settings/email-folders",
    "/settings/extensions",
}

@app.middleware("http")
async def rate_limiter_mw(request: Request, call_next):
    # Exempter les requêtes OPTIONS (CORS preflight) du rate limiting
    if request.method == "OPTIONS":
        return await call_next(request)

    if request.method == "GET" and request.url.path in RATE_LIMIT_EXEMPT_GET_PATHS:
        return await call_next(request)
    
    client_ip = request.client.host if request.client else "unknown"
    if not RATE_LIMITER.allow(client_ip):
        return JSONResponse({"detail": "Trop de requêtes, réessaie dans quelques secondes."}, status_code=429)
    return await call_next(request)

init_db()

# Endpoints config
app.include_router(config_router)
app.include_router(orchestrator_debug_router)
app.include_router(response_trace_router)

# Paramètres → E-mails, Dossiers, Extensions
app.include_router(email_settings_router)
app.include_router(directory_settings_router)
app.include_router(extensions_settings_router)  # ✅ nouveau
app.include_router(chats_router)  # ✅ nouveau: chats persistants
app.include_router(projects_router)
app.include_router(mail_sync_router)  # ✅ nouveau: mail sync (delta-like)

# Reste de l'API
app.include_router(status_router)
app.include_router(ingest_router)
app.include_router(ask_router)
app.include_router(sources_router)

@app.on_event("startup")
async def on_startup():
    """Start the auto-sync scheduler on application startup."""
    start_auto_sync_scheduler()

@app.on_event("shutdown")
async def on_shutdown():
    """Stop the auto-sync scheduler gracefully on application shutdown."""
    stop_auto_sync_scheduler()
