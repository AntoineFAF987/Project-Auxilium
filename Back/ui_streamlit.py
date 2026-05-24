# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 23:40:25 2025

@author: aejau
"""

# -*- coding: utf-8 -*-
"""
UI Streamlit — chat + rendu math LaTeX (robuste)
"""

import re
import requests
import streamlit as st
from streamlit import components
from streamlit.errors import StreamlitAPIException

# =========================
# Page config (doit être le 1er st.* ; tolérant si déjà appelé)
# =========================
try:
    st.set_page_config(page_title="RAG on‑prem — Chat", layout="wide")
except StreamlitAPIException:
    pass  # déjà appelé ailleurs

API = "http://127.0.0.1:8765"
st.title("🧠 RAG on‑prem — Chat")

# --- CSS ---
st.markdown("""
<style>
.bubble {padding:0.8rem 1rem; border-radius:12px; box-shadow:0 1px 3px rgba(0,0,0,0.06); max-width:900px; word-wrap:break-word;}
.bubble-user {background:#e8f0fe;}
.bubble-assistant {background:#f5f5f5;}
.meta {font-size:0.78rem; color:#6b7280; margin-top:0.25rem;}
.chat-wrap {display:flex; flex-direction:column; height:calc(100vh - 180px);}
.chat-scroll {flex:1; overflow:auto; padding-bottom:0.5rem;}
html, body, .block-container {caret-color: transparent;}
textarea[data-testid="stChatInputTextArea"] {caret-color: auto;}
.bubble:focus {outline:none;}
</style>
""", unsafe_allow_html=True)

# --- MathJax (rendu LaTeX) ---
components.v1.html("""
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
""", height=0)

# ---------- Helpers LaTeX ----------
# CORRIGE: parenthèses correctement équilibrées
_math_inline = re.compile(r"\\\((.+?)\\\)", flags=re.S)   # \( ... \)
_math_block  = re.compile(r"\\\[(.+?)\\\]", flags=re.S)   # \[ ... \]

def _texify(s: str) -> str:
    """Convertit \(..\)->$..$ et \[..]->$$..$$, sans toucher aux blocs ```code```."""
    if not s:
        return s
    parts = re.split(r"(```.*?```)", s, flags=re.S)  # protège les blocs code
    for i in range(0, len(parts), 2):
        chunk = parts[i]
        chunk = _math_block.sub(r"$$\\1$$", chunk)
        chunk = _math_inline.sub(r"$\\1$", chunk)
        parts[i] = chunk
    return "".join(parts)

def _mathjax_typeset():
    components.v1.html("""
    <script>
      if (window.MathJax && MathJax.typesetPromise) { MathJax.typesetPromise(); }
    </script>
    """, height=0)

# ---------- Sidebar (admin) ----------
with st.sidebar:
    st.header("Admin")
    c1, c2, c3 = st.columns(3)
    if c1.button("Status"):
        try:
            st.json(requests.get(f"{API}/status", timeout=30).json())
        except Exception as e:
            st.error(str(e))
    if c2.button("Reindex"):
        try:
            r = requests.post(f"{API}/reindex", timeout=1800)
            st.success("Reindex OK" if r.ok else r.text)
        except Exception as e:
            st.error(str(e))
    if c3.button("Vider chat"):
        st.session_state.messages = []
        st.experimental_rerun()

    st.divider()
    c4, c5 = st.columns(2)
    if c4.button("Ingest Emails"):
        try:
            r = requests.post(f"{API}/ingest_emails", timeout=1800)
            st.success("Emails OK" if r.ok else r.text)
        except Exception as e:
            st.error(str(e))
    if c5.button("Ingest PDFs"):
        try:
            r = requests.post(f"{API}/ingest_pdfs", timeout=1800)
            st.success("PDFs OK" if r.ok else r.text)
        except Exception as e:
            st.error(str(e))

# ---------- Historique ----------
if "messages" not in st.session_state:
    # [{role:"user"|"assistant", content:"...", sources:[...]?}]
    st.session_state.messages = []

# ---------- Conteneur défilant ----------
st.markdown('<div class="chat-wrap"><div id="chat-scroll" class="chat-scroll">', unsafe_allow_html=True)

def render_message(role: str, content: str, sources=None):
    sources = sources or []
    left, right = st.columns([6, 6])
    if role == "assistant":
        with left:
            content_tx = _texify(content)  # <<< rendu latex
            st.markdown(f'<div class="bubble bubble-assistant">{content_tx}</div>', unsafe_allow_html=True)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        path = s.get("path", "?")
                        chunk = s.get("chunk", "?")
                        st.markdown(f"- `{path}` (chunk {chunk})")
            _mathjax_typeset()
    else:
        with right:
            st.markdown(
                f'<div style="text-align:right;"><div class="bubble bubble-user">{content}</div></div>',
                unsafe_allow_html=True
            )

# Affichage historique
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("sources"))

# Ancre + auto-scroll
st.markdown('<div id="chat-end"></div></div></div>', unsafe_allow_html=True)
components.v1.html("""
<script>
  const el = window.parent.document.querySelector('#chat-scroll');
  if (el) { el.scrollTop = el.scrollHeight; }
</script>
""", height=0)

# ---------- Saisie ----------
prompt = st.chat_input("Écris ton message… (Entrée pour envoyer)")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message("user", prompt)

    with st.spinner("Réflexion en cours…"):
        try:
            r = requests.post(f"{API}/ask", json={"q": prompt}, timeout=120)
            if not r.ok:
                st.error(r.text or "Erreur API")
            else:
                data = r.json()
                answer = (data.get("answer") or "").strip() or "_(pas de réponse)_"
                sources = data.get("sources") or []
                render_message("assistant", answer, sources)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
        except Exception as e:
            st.error(str(e))

# Re-scroll en bas après envoi
components.v1.html("""
<script>
  const el = window.parent.document.querySelector('#chat-scroll');
  if (el) { el.scrollTop = el.scrollHeight; }
</script>
""", height=0)
