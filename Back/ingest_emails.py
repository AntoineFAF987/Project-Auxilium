# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 12:32:11 2025

@author: aejau
"""

# ingest_emails.py
# --- Ingestion d'emails via Microsoft Graph ---
# Cache persistant (par dossier/utilisateur) pour éviter de recharger les mêmes mails
# - Mode delegated (device flow) OU via access token fourni par le front
# - Export JSON + pièces jointes + flatten .txt pour RAG
# - Cache: sync_state.json (last_sync + processed_ids) par (mode+user+folder)

import os, json, re, html
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

import requests
import msal

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except Exception:
    HAS_BS4 = False

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
DEFAULT_TOP = 50  # page size
RESERVED_SCOPES = {"openid", "profile", "offline_access"}

# =========================
# Utils (fichiers, temps, JSON)
# =========================
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _read_json(path: str, default=None):
    if not os.path.exists(path):
        return {} if default is None else default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {} if default is None else default

def _write_json(path: str, obj: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _resolve_config_path(config_path: str, configured_path: str) -> str:
    """Resolve an ingestion path relative to config.json, never to process CWD."""
    if os.path.isabs(configured_path):
        return os.path.normpath(configured_path)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.normpath(os.path.join(config_dir, configured_path))

def _resolved_email_config(config_path: str, email_config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(email_config)
    for key in ("output_dir", "attachments_dir", "flatten_for_rag_dir"):
        value = resolved.get(key)
        if value:
            resolved[key] = _resolve_config_path(config_path, value)
    return resolved

def _utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def _max_iso(a: Optional[str], b: Optional[str]) -> Optional[str]:
    da, db = _parse_iso(a), _parse_iso(b)
    if da and db:
        return a if da >= db else b
    return a or b

# =========================
# Normalisation / debug Graph
# =========================
def _normalize_scopes(cfg_scopes: Optional[List[str]]) -> List[str]:
    scopes = []
    for s in (cfg_scopes or []):
        if s in RESERVED_SCOPES:
            continue
        scopes.append(s if s.startswith("https://") else f"https://graph.microsoft.com/{s}")
    return scopes or ["https://graph.microsoft.com/User.Read", "https://graph.microsoft.com/Mail.Read"]

def _graph_get(url: str, headers: Dict[str, str], timeout: int = 30) -> Dict[str, Any]:
    resp = requests.get(url, headers=headers, timeout=timeout)
    if resp.status_code >= 400:
        wa = resp.headers.get("WWW-Authenticate")
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        print(f"\n--- Graph error {resp.status_code} on {url} ---")
        if wa:
            print(f"WWW-Authenticate: {wa}")
        print(body)
        print("--- end ---")
    resp.raise_for_status()
    return resp.json()

def _graph_paged(url: str, headers: Dict[str, str], timeout: int = 30):
    while url:
        data = _graph_get(url, headers, timeout=timeout)
        yield data
        url = data.get("@odata.nextLink")

# =========================
# Auth MSAL (device flow, pour fallback)
# =========================
def _get_headers_delegated(authority: str, client_id: str, scopes: List[str], cache_path: str) -> Dict[str, str]:
    cache_path = os.path.abspath(cache_path)
    _ensure_dir(os.path.dirname(cache_path))
    cache = msal.SerializableTokenCache()
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                cache.deserialize(cache_file.read())
        except Exception as exc:
            raise RuntimeError(f"Cache MSAL illisible: {cache_path}") from exc
    app = msal.PublicClientApplication(client_id=client_id, authority=authority, token_cache=cache)

    def persist_cache_if_changed() -> None:
        if not cache.has_state_changed:
            return
        tmp_path = cache_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as cache_file:
                cache_file.write(cache.serialize())
            os.replace(tmp_path, cache_path)
        except Exception as exc:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            raise RuntimeError(f"Impossible de persister le cache MSAL: {cache_path}") from exc

    # Try every cached account before deciding that interactive auth is needed.
    for account in app.get_accounts():
        res = app.acquire_token_silent(scopes, account=account)
        # Silent renewal may rotate a refresh token and mutate the cache.
        persist_cache_if_changed()
        if res and "access_token" in res:
            print("🔑 Token (silent) scopes:", res.get("scope") or res.get("scopes"))
            print("🔗 Authority:", authority)
            return {"Authorization": f"Bearer {res['access_token']}"}

    flow = app.initiate_device_flow(scopes=scopes)
    if "user_code" not in flow:
        raise RuntimeError("Échec du device flow (vérifie client_id / scopes).")
    print("\n===== Authentification requise =====")
    print(flow["message"])
    res = app.acquire_token_by_device_flow(flow)
    if "access_token" not in res:
        raise RuntimeError(f"Échec token: {res.get('error_description')}")
    persist_cache_if_changed()
    print("🔑 Token scopes:", res.get("scope") or res.get("scopes"))
    print("🔗 Authority:", authority)
    return {"Authorization": f"Bearer {res['access_token']}"}

def _get_headers_application(authority: str, client_id: str, client_secret: str) -> Dict[str, str]:
    import msal
    app = msal.ConfidentialClientApplication(
        client_id=client_id,
        client_credential=client_secret,
        authority=authority
    )
    res = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
    if "access_token" not in res:
        raise RuntimeError(f"Token app KO: {res.get('error_description') or res}")
    print("🔑 App Token (scope=.default) OK")
    print("🔗 Authority:", authority)
    return {"Authorization": f"Bearer {res['access_token']}"}

# =========================
# URLs selon mode (me/users)
# =========================
def _folders_url(user: Optional[str]) -> str:
    return f"{GRAPH_BASE}/me/mailFolders" if not user else f"{GRAPH_BASE}/users/{user}/mailFolders"

def _messages_url(user: Optional[str], folder_id: Optional[str]) -> str:
    if not user:
        return f"{GRAPH_BASE}/me/messages" if not folder_id else f"{GRAPH_BASE}/me/mailFolders/{folder_id}/messages"
    return f"{GRAPH_BASE}/users/{user}/messages" if not folder_id else f"{GRAPH_BASE}/users/{user}/mailFolders/{folder_id}/messages"

def _attachments_url(user: Optional[str], message_id: str) -> str:
    return f"{GRAPH_BASE}/me/messages/{message_id}/attachments" if not user else f"{GRAPH_BASE}/users/{user}/messages/{message_id}/attachments"

# =========================
# Folders & messages
# =========================
def list_mail_folders(headers: Dict[str, str], user: Optional[str]) -> List[Dict[str, Any]]:
    url = _folders_url(user) + "?$top=200&$select=id,displayName"
    return [f for page in _graph_paged(url, headers) for f in page.get("value", [])]

def fetch_folder_id(headers: Dict[str, str], user: Optional[str], folder_name: str) -> Optional[str]:
    WK = {
        "inbox": "inbox",
        "boîte de réception": "inbox",
        "boite de reception": "inbox",
        "sent items": "sentitems",
        "éléments envoyés": "sentitems",
        "elements envoyes": "sentitems",
        "drafts": "drafts",
        "brouillons": "drafts",
        "junk": "junkemail",
        "courrier indésirable": "junkemail",
        "courrier indesirable": "junkemail",
        "deleted items": "deleteditems",
        "éléments supprimés": "deleteditems",
        "elements supprimes": "deleteditems",
        "archive": "archive",
        "outbox": "outbox"
    }
    key = (folder_name or "").lower()
    if key in WK:
        url = (f"{_folders_url(user)}/{WK[key]}?$select=id,displayName")
        try:
            data = _graph_get(url, headers, timeout=30)
            return data.get("id")
        except Exception:
            pass  # fallback libellé

    url = _folders_url(user) + "?$top=200&$select=id,displayName"
    for page in _graph_paged(url, headers, timeout=30):
        for f in page.get("value", []):
            if (f.get("displayName") or "").lower() == key:
                return f.get("id")
    return None

def fetch_messages(headers: Dict[str, str], user: Optional[str], folder_id: Optional[str],
                   date_since_iso: Optional[str], max_emails: int) -> List[Dict[str, Any]]:
    base = _messages_url(user, folder_id)
    params = [
        "$select=id,subject,from,toRecipients,ccRecipients,receivedDateTime,conversationId,categories,hasAttachments,body",
        "$orderby=receivedDateTime desc",
        "$top=" + str(min(DEFAULT_TOP, max_emails))
    ]
    if date_since_iso:
        params.append(f"$filter=receivedDateTime ge {date_since_iso}")

    url = base + "?" + "&".join(params)
    print(f"  ↪︎ GET {url}")
    results, page = [], 0

    while url and len(results) < max_emails:
        data = _graph_get(url, headers, timeout=40)
        items = data.get("value", [])
        page += 1
        print(f"  • page {page}: +{len(items)}")
        results.extend(items)
        url = data.get("@odata.nextLink")
        if not items:
            break

    print(f"  ➜ total récupérés (avant coupe): {len(results)}")
    return results[:max_emails]

def fetch_attachments(headers: Dict[str, str], user: Optional[str], message_id: str) -> List[Dict[str, Any]]:
    url = _attachments_url(user, message_id) + "?$top=50"
    data = _graph_get(url, headers, timeout=30)
    return data.get("value", [])

# =========================
# Sauvegarde locale
# =========================
def _normalize_recipients(recips):
    if not recips: return []
    out = []
    for r in recips:
        addr = r.get("emailAddress") or {}
        out.append({"name": addr.get("name"), "address": addr.get("address")})
    return out

def _safe_name(s: str) -> str:
    return re.sub(r"[^\w\-]+", "_", (s or "no_subject"))

def html_to_text(html_content: str) -> str:
    if not html_content:
        return ""
    if HAS_BS4:
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    text = re.sub(r"<br\s*/?>", "\n", html_content, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()

def save_attachment(att: Dict[str, Any], dest_dir: str, prefix: str) -> Optional[str]:
    _ensure_dir(dest_dir)
    name = att.get("name") or "attachment"
    safe = re.sub(r"[^\w\-. ]", "_", name)
    path = os.path.join(dest_dir, f"{prefix}__{safe}")
    # Graph returns ``#microsoft.graph.fileAttachment``.  Casing is not a
    # contract, so normalize it before deciding whether binary content exists.
    if str(att.get("@odata.type") or "").casefold().endswith("fileattachment"):
        import base64
        b64 = att.get("contentBytes")
        if b64:
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            return path
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(att, f, ensure_ascii=False, indent=2)
    return path + ".json"

def flatten_email_for_rag(email_obj: Dict[str, Any]) -> str:
    hdr = []
    hdr.append(f"Subject: {email_obj.get('subject') or ''}")
    frm = email_obj.get("from_") or {}
    hdr.append(f"From: {frm.get('name','')} <{frm.get('address','')}>")
    to_list = ", ".join([f"{r.get('name','')} <{r.get('address','')}>" for r in email_obj.get("to",[])])
    cc_list = ", ".join([f"{r.get('name','')} <{r.get('address','')}>" for r in email_obj.get("cc",[])])
    hdr.append(f"To: {to_list}")
    if cc_list: hdr.append(f"CC: {cc_list}")
    hdr.append(f"Date: {email_obj.get('receivedDateTime','')}")
    if email_obj.get("categories"): hdr.append(f"Categories: {', '.join(email_obj['categories'])}")
    body_text = email_obj.get("body_text") or ""
    return ("\n".join(hdr) + "\n\n" + body_text).strip() + "\n"

# =========================
# Ingestion — mode historique (device flow ou application)
# =========================
def ingest_emails(config_path: str, override_folders: Optional[List[str]] = None):
    cfg = _read_json(config_path)
    graph = cfg["graph"]
    E = _resolved_email_config(config_path, cfg["email_ingest"])

    out_dir  = E["output_dir"]
    att_dir  = E["attachments_dir"]
    flat_dir = E["flatten_for_rag_dir"]
    _ensure_dir(out_dir); _ensure_dir(att_dir); _ensure_dir(flat_dir)

    state_path = os.path.join(out_dir, "sync_state.json")
    state = _read_json(state_path, default={})

    auth_mode = (graph.get("auth_mode") or "delegated").lower().strip()
    authority = graph.get("authority") or "https://login.microsoftonline.com/consumers"
    client_id = graph["client_id"]

    # --- Auth & headers ---
    if auth_mode == "application":
        client_secret = graph.get("client_secret")
        if not client_secret:
            raise RuntimeError("auth_mode=application mais client_secret manquant dans graph.")
        headers = _get_headers_application(authority, client_id, client_secret)
        target_users = E.get("target_users") or []
        if not target_users:
            raise RuntimeError("auth_mode=application : email_ingest.target_users doit contenir au moins une boîte (UPN).")
    else:
        scopes = _normalize_scopes(graph.get("scopes"))
        cache_path = os.path.join(out_dir, "token_cache.bin")
        headers = _get_headers_delegated(authority, client_id, scopes, cache_path)
        target_users = [None]  # /me en delegated

    # --- Self-test + UPN pour clé de cache
    try:
        if auth_mode == "application":
            who = (E.get("target_users") or ["unknown"])[0]
            me_like = _graph_get(f"{GRAPH_BASE}/users/{who}?$select=id,displayName,mail,userPrincipalName", headers)
            upn = me_like.get("userPrincipalName") or me_like.get("mail") or who
        else:
            me = _graph_get(f"{GRAPH_BASE}/me?$select=id,displayName,mail,userPrincipalName", headers)
            upn = me.get("userPrincipalName") or me.get("mail") or "me"
        print(f"👤 Compte visé: {upn}")
    except Exception:
        print("⚠️ Test accès KO. Abandon.")
        return

    total_saved = 0
    # Détermine la liste des dossiers à traiter.  On n’utilise plus du tout
    # config.json pour sélectionner les boîtes mail : seule la valeur
    # ``override_folders`` est prise en compte.  Si la liste est vide ou
    # absente, l’ingestion est abandonnée.
    folders_wanted = list(override_folders or [])
    if not folders_wanted:
        print(
            "ℹ️  Aucun dossier sélectionné pour l’ingestion emails → opération annulée."
        )
        return
    global_default_since = E.get("date_since")  # ISO

    for user in target_users:
        user_label = upn if user is None else (user or "unknown")
        for folder_name in folders_wanted:
            print(f"\n📥 Dossier ({user_label}): {folder_name}")
            fid = fetch_folder_id(headers, user, folder_name)
            if fid is None:
                print(f"  ⚠️ Dossier introuvable: {folder_name}")
                try:
                    names = [f.get("displayName") for f in list_mail_folders(headers, user) if f.get("displayName")]
                    if names:
                        print("  📂 Dossiers disponibles (extraits): " + ", ".join(names[:30]) + (" ..." if len(names) > 30 else ""))
                except Exception:
                    pass
                print("  → On passe au dossier suivant.")
                continue

            skey = f"legacy:{user_label}:{folder_name.lower()}"
            entry = state.get(skey, {})
            last_sync = entry.get("last_sync")
            processed_ids = set(entry.get("processed_ids", []))
            effective_since = _max_iso(global_default_since, last_sync)
            if effective_since:
                print(f"  🗂 Cache: since={effective_since} | déjà vus={len(processed_ids)}")
            else:
                print(f"  🗂 Cache: première synchro")

            msgs = fetch_messages(
                headers=headers, user=user, folder_id=fid,
                date_since_iso=effective_since, max_emails=E.get("max_emails", 1000)
            )
            print(f"  ➜ {len(msgs)} messages retournés (avant dédup)")

            new_count = 0
            for m in msgs:
                mid = m.get("id")
                if not mid or mid in processed_ids:
                    continue

                subject  = (m.get("subject") or "").strip()
                received = m.get("receivedDateTime")
                from_    = (m.get("from") or {}).get("emailAddress") or {}
                body     = (m.get("body") or {})
                ctype    = (body.get("contentType") or "").lower()
                raw      = body.get("content") or ""

                if ctype == "html":
                    body_text = html_to_text(raw); body_html = raw
                else:
                    body_text = raw; body_html = None

                email_obj = {
                    "id": mid,
                    "subject": subject,
                    "from_": {"name": from_.get("name"), "address": from_.get("address")},
                    "to": _normalize_recipients(m.get("toRecipients")),
                    "cc": _normalize_recipients(m.get("ccRecipients")),
                    "receivedDateTime": received,
                    "conversationId": m.get("conversationId"),
                    "categories": m.get("categories") or [],
                    "hasAttachments": bool(m.get("hasAttachments")),
                    "body_text": body_text,
                    "body_html": body_html if E.get("include_body_html", False) else None,
                    "user": user_label,
                    "folder": folder_name
                }

                att_paths = []
                if email_obj["hasAttachments"]:
                    try:
                        for i, att in enumerate(fetch_attachments(headers, user, mid)):
                            p = save_attachment(att, E["attachments_dir"], prefix=(mid[:8] + f"_{i:02d}"))
                            if p: att_paths.append(p)
                    except Exception as ex:
                        print(f"    ⚠️ Attachements KO ({subject[:40]}): {ex}")
                email_obj["attachment_paths"] = att_paths

                fname = _safe_name(subject)[:80]
                date_prefix = (received[:10] if received else "nodate")
                json_path = os.path.join(E["output_dir"], f"{date_prefix}__{fname}__{(mid or '')[:8]}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(email_obj, f, ensure_ascii=False, indent=2)

                flat_txt = flatten_email_for_rag(email_obj)
                flat_path = os.path.join(E["flatten_for_rag_dir"], f"{date_prefix}__{fname}__{(mid or '')[:8]}.txt")
                with open(flat_path, "w", encoding="utf-8") as f:
                    f.write(flat_txt)

                processed_ids.add(mid)
                new_count += 1
                total_saved += 1

            entry["last_sync"] = _utc_now_iso()
            MAX_IDS = 100_000
            entry["processed_ids"] = list(sorted(processed_ids))[-MAX_IDS:] if len(processed_ids) > MAX_IDS else list(processed_ids)
            state[skey] = entry
            _write_json(state_path, state)

            print(f"  ✅ Nouveaux ingérés: {new_count} (cumul cache vus={len(entry['processed_ids'])})")

    print(f"\n✅ Ingestion emails terminée. Nouveaux enregistrements cette session: {total_saved}")
    print(f"   JSON: {E['output_dir']}")
    print(f"   TXT (RAG): {E['flatten_for_rag_dir']}")
    print(f"   Pièces jointes: {E['attachments_dir']}")

# =========================
# Ingestion — via access token du FRONT (toujours /me)
# =========================
def ingest_emails_with_access_token(
    config_path: str,
    access_token: str,
    override_folders: Optional[List[str]] = None,
) -> None:
    """
    Ingestion pour la boîte du compte réellement connecté côté FRONT.

    :param config_path: chemin vers le fichier ``config.json`` (graph/email_ingest)
    :param access_token: jeton Bearer reçu du front (oauth2)
    :param override_folders: liste facultative de noms de dossiers à ingérer.
        Si fournie, **seule cette liste** est utilisée.  Si la liste est vide
        ou absente, **aucune ingestion** n'est réalisée.  Aucune lecture de
        ``email_ingest.folders`` n'est effectuée.

    L'utilisateur est toujours ``/me`` en mode token issu du front.  Le cache est
    isolé par utilisateur (clé ``front:<UPN>:<folder>``).
    """
    if not access_token or not access_token.strip():
        raise RuntimeError("Access token manquant pour ingest_emails_with_access_token")

    cfg = _read_json(config_path)
    E = _resolved_email_config(config_path, cfg["email_ingest"])

    out_dir  = E["output_dir"]
    att_dir  = E["attachments_dir"]
    flat_dir = E["flatten_for_rag_dir"]
    _ensure_dir(out_dir); _ensure_dir(att_dir); _ensure_dir(flat_dir)

    state_path = os.path.join(out_dir, "sync_state.json")
    state = _read_json(state_path, default={})

    headers = {"Authorization": f"Bearer {access_token}"}

    # Qui est /me ?
    me = _graph_get(f"{GRAPH_BASE}/me?$select=id,displayName,mail,userPrincipalName", headers)
    upn = me.get("userPrincipalName") or me.get("mail") or "me"
    print(f"👤 FRONT access token → boîte: {upn}")

    total_saved = 0
    # Ne lit plus config.json pour les dossiers : seules les valeurs
    # ``override_folders`` comptent.  Si absentes ou vides, l’ingestion est
    # abandonnée.  Ce comportement est cohérent avec l’utilisation de la base
    # locale pour la sélection des dossiers.
    folders_wanted = list(override_folders or [])
    if not folders_wanted:
        print(
            "ℹ️  Aucun dossier sélectionné pour l’ingestion emails → opération annulée."
        )
        return
    global_default_since = E.get("date_since")  # ISO

    user = None  # /me
    for folder_name in folders_wanted:
        print(f"\n📥 Dossier ({upn}): {folder_name}")
        fid = fetch_folder_id(headers, user, folder_name)
        if fid is None:
            print(f"  ⚠️ Dossier introuvable: {folder_name}")
            try:
                names = [f.get("displayName") for f in list_mail_folders(headers, user) if f.get("displayName")]
                if names:
                    print("  📂 Dossiers disponibles (extraits): " + ", ".join(names[:30]) + (" ..." if len(names) > 30 else ""))
            except Exception:
                pass
            print("  → On passe au dossier suivant.")
            continue

        # ⚠️ Clé de cache dédiée au user FRONT (évite mélange avec anciens comptes)
        skey = f"front:{upn}:{folder_name.lower()}"
        entry = state.get(skey, {})
        last_sync = entry.get("last_sync")
        processed_ids = set(entry.get("processed_ids", []))
        effective_since = _max_iso(global_default_since, last_sync)
        if effective_since:
            print(f"  🗂 Cache: since={effective_since} | déjà vus={len(processed_ids)}")
        else:
            print(f"  🗂 Cache: première synchro")

        msgs = fetch_messages(
            headers=headers, user=user, folder_id=fid,
            date_since_iso=effective_since, max_emails=E.get("max_emails", 1000)
        )
        print(f"  ➜ {len(msgs)} messages retournés (avant dédup)")

        new_count = 0
        for m in msgs:
            mid = m.get("id")
            if not mid or mid in processed_ids:
                continue

            subject  = (m.get("subject") or "").strip()
            received = m.get("receivedDateTime")
            from_    = (m.get("from") or {}).get("emailAddress") or {}
            body     = (m.get("body") or {})
            ctype    = (body.get("contentType") or "").lower()
            raw      = body.get("content") or ""

            if ctype == "html":
                body_text = html_to_text(raw); body_html = raw
            else:
                body_text = raw; body_html = None

            email_obj = {
                "id": mid,
                "subject": subject,
                "from_": {"name": from_.get("name"), "address": from_.get("address")},
                "to": _normalize_recipients(m.get("toRecipients")),
                "cc": _normalize_recipients(m.get("ccRecipients")),
                "receivedDateTime": received,
                "conversationId": m.get("conversationId"),
                "categories": m.get("categories") or [],
                "hasAttachments": bool(m.get("hasAttachments")),
                "body_text": body_text,
                "body_html": body_html if E.get("include_body_html", False) else None,
                "user": upn,
                "folder": folder_name
            }

            att_paths = []
            if email_obj["hasAttachments"]:
                try:
                    for i, att in enumerate(fetch_attachments(headers, user, mid)):
                        p = save_attachment(att, E["attachments_dir"], prefix=(mid[:8] + f"_{i:02d}"))
                        if p: att_paths.append(p)
                except Exception as ex:
                    print(f"    ⚠️ Attachements KO ({subject[:40]}): {ex}")
            email_obj["attachment_paths"] = att_paths

            fname = _safe_name(subject)[:80]
            date_prefix = (received[:10] if received else "nodate")
            json_path = os.path.join(E["output_dir"], f"{date_prefix}__{fname}__{(mid or '')[:8]}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(email_obj, f, ensure_ascii=False, indent=2)

            flat_txt = flatten_email_for_rag(email_obj)
            flat_path = os.path.join(E["flatten_for_rag_dir"], f"{date_prefix}__{fname}__{(mid or '')[:8]}.txt")
            with open(flat_path, "w", encoding="utf-8") as f:
                f.write(flat_txt)

            processed_ids.add(mid)
            new_count += 1
            total_saved += 1

        entry["last_sync"] = _utc_now_iso()
        MAX_IDS = 100_000
        entry["processed_ids"] = list(sorted(processed_ids))[-MAX_IDS:] if len(processed_ids) > MAX_IDS else list(processed_ids)
        state[skey] = entry
        _write_json(state_path, state)

        print(f"  ✅ Nouveaux ingérés: {new_count} (cumul cache vus={len(entry['processed_ids'])})")

    print(f"\n✅ Ingestion (front) terminée pour {upn}. Nouveaux enregistrements: {total_saved}")
    print(f"   JSON: {E['output_dir']}")
    print(f"   TXT (RAG): {E['flatten_for_rag_dir']}")
    print(f"   Pièces jointes: {E['attachments_dir']}")
