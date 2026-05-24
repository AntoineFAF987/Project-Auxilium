# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 19:34:59 2025

@author: aejau
"""

# ingest_pdfs.py
import os, re
from typing import List
import pdfplumber

try:
    import fitz
    HAVE_FITZ = True
except Exception:
    HAVE_FITZ = False

# --- Config PDF & chunking (tu peux garder ces valeurs) ---
VERBOSE_PDF = True
MAX_PDF_SIZE_MB = 60
HARD_LIMIT_PAGES = 50
MAX_TEXT_CHARS_PER_PDF = 2_000_000
CHUNK_CHARS = 800
CHUNK_OVERLAP = 200

SENT_SEP_RE = re.compile(r'([\.!?])(\s+)')

def split_text_fast(txt: str, max_chars=CHUNK_CHARS, overlap=CHUNK_OVERLAP) -> List[str]:
    # ⇩ copie exacte de ta fonction
    if len(txt) <= max_chars:
        return [txt]
    parts, last = [], 0
    for m in SENT_SEP_RE.finditer(txt):
        end = m.end()
        parts.append(txt[last:end]); last = end
    if last < len(txt):
        parts.append(txt[last:])
    chunks, buf, buf_len = [], [], 0
    step_fixed = max_chars - overlap if overlap < max_chars else max_chars
    def flush():
        nonlocal buf, buf_len
        if not buf: return
        c = "".join(buf).strip()
        if c: chunks.append(c)
        if overlap > 0 and chunks:
            tail = chunks[-1][-overlap:]; buf = [tail]; buf_len = len(tail)
        else:
            buf, buf_len = [], 0
    for p in parts:
        p = (p or "").lstrip()
        if buf_len + len(p) <= max_chars:
            buf.append(p); buf_len += len(p)
        else:
            flush()
            if len(p) > max_chars:
                start = 0
                while start < len(p):
                    end = min(start + max_chars, len(p))
                    chunk_slice = p[start:end]
                    if chunk_slice: chunks.append(chunk_slice)
                    if end >= len(p):
                        if overlap > 0:
                            tail = p[max(0, end - overlap):end]
                            buf = [tail]; buf_len = len(tail)
                        else:
                            buf, buf_len = [], 0
                        break
                    start += step_fixed
            else:
                buf = [p]; buf_len = len(p)
    flush()
    if not chunks:
        start = 0
        while start < len(txt):
            end = min(start + max_chars, len(txt))
            chunks.append(txt[start:end])
            if end >= len(txt): break
            start += step_fixed
    return chunks

def read_pdf(path: str) -> str:
    # ⇩ copie exacte de ta fonction (PyMuPDF puis pdfplumber)
    if HAVE_FITZ:
        try:
            doc = fitz.open(path)
            if doc.is_encrypted:
                try: doc.authenticate("")
                except Exception:
                    print(f"🔒 PDF chiffré (PyMuPDF) → skip: {path}")
                    doc.close(); return ""
            texts = []
            page_count = len(doc)
            limit = page_count if HARD_LIMIT_PAGES <= 0 else min(page_count, HARD_LIMIT_PAGES)
            for i in range(limit):
                if VERBOSE_PDF and (i % 10 == 0 or i == limit - 1):
                    print(f"   • page {i+1}/{limit}")
                try: t = doc[i].get_text("text") or ""
                except Exception as e:
                    print(f"   ⚠️  Erreur page {i+1}: {e}"); t = ""
                if t.strip(): texts.append(t.strip())
            doc.close()
            return "\n\n".join(texts)
        except Exception as e:
            print(f"ℹ️  PyMuPDF a échoué ({e}), on tente pdfplumber…")
    try:
        with pdfplumber.open(path) as pdf:
            if getattr(pdf, "is_encrypted", False):
                try: pdf.decrypt("")
                except Exception:
                    print(f"🔒 PDF chiffré (pdfplumber) → skip: {path}")
                    return ""
            texts = []
            page_count = len(pdf.pages)
            limit = page_count if HARD_LIMIT_PAGES <= 0 else min(page_count, HARD_LIMIT_PAGES)
            for i in range(limit):
                if VERBOSE_PDF and (i % 10 == 0 or i == limit - 1):
                    print(f"   • page {i+1}/{limit}")
                try: t = pdf.pages[i].extract_text() or ""
                except Exception as e:
                    print(f"   ⚠️  Erreur page {i+1}: {e}"); t = ""
                if t.strip(): texts.append(t.strip())
            return "\n\n".join(texts)
    except Exception as e:
        print(f"❌ Échec pdfplumber sur {path}: {e}")
        return ""
