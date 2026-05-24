# -*- coding: utf-8 -*-
"""
Utils & tokenisation — version complète.
- Tokenisation simple (BM25-like) + stopwords FR
- Utilitaires fichiers / parcours
- Liste d'extensions supportées (modifiable à chaud par l'API)
"""

import os
import json
import hashlib
import re
import unicodedata
import fnmatch
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from typing import List, Dict, Tuple, Iterable, Optional, Set

# =========================
# Stopwords FR & tokenisation BM25
# =========================
FRENCH_STOPWORDS = {
    "le","la","les","de","des","du","un","une","et","ou","au","aux","en","à","dans","sur",
    "par","pour","avec","sans","ce","cet","cette","ces","il","elle","ils","elles","on",
    "nous","vous","je","tu","se","sa","son","ses","leur","leurs","d","l","j","t","qu",
    "que","qui","quoi","dont","où","mais","car","ni","or","donc","ne","pas","plus","moins",
    "aujourd","hui","comme","ainsi","afin","entre","vers","sous","chez","tout","tous","toute","toutes",
    "fait","faut","avoir","être","ete","sont","est","sera",
}

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def tokenize_for_bm25(text: str) -> List[str]:
    text = _strip_accents(text.lower())
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if t and t not in FRENCH_STOPWORDS]

def keyword_overlap_count(question: str, context: str) -> int:
    if not context:
        return 0
    q_tokens = set(tokenize_for_bm25(question))
    if not q_tokens:
        return 0
    c_tokens = set(tokenize_for_bm25(context))
    return len(q_tokens & c_tokens)

# =========================
# Utils fichiers / dossiers
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def file_fingerprint(path: str) -> str:
    """
    Empreinte (path+taille+mtime) — utile pour détecter rapidement un changement.
    """
    try:
        st = os.stat(path)
        payload = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    except Exception:
        payload = f"{path}|0|0"
    return hashlib.md5(payload.encode()).hexdigest()

def _is_under(path: str, root: str) -> bool:
    """
    Vrai si 'path' est sous 'root' (prévention sorties non désirées).
    """
    try:
        return os.path.abspath(path).startswith(os.path.abspath(root))
    except Exception:
        return False

def _safe_walk(root: str, followlinks: bool = False) -> Iterable[Tuple[str, list, list]]:
    """
    Parcours robuste (ignore silencieusement certaines erreurs d'accès).
    """
    try:
        for t in os.walk(root, followlinks=followlinks):
            yield t
    except Exception:
        return

# =========================
# Extensions supportées (modifiable à chaud)
# =========================
# Par défaut : un ensemble large d’extensions "texte ou texte-extractible".
# - Fichiers texte / config / data : lecture directe (UTF-8 errors='replace')
# - Code : idem (utile pour RAG sur base de code)
# - Bureautique modernes : .pdf (extracteur dédié ailleurs), .docx/.pptx/.xlsx (best-effort texte)
# - Les anciens formats binaires (.doc/.ppt/.xls) et OneNote (.one) NE SONT PAS inclus par défaut.
SUPPORTED_EXTS: Set[str] = {
    # Documents texte
    ".txt", ".md", ".rst", ".rtf",  # (RTF lu comme texte brut ; si binaire exotique, sera ignoré au parsing)
    # Données / config textuelles
    ".json", ".jsonl", ".csv", ".tsv", ".xml", ".ini", ".toml", ".cfg", ".conf", ".log",
    ".yml", ".yaml",
    # Web / markup
    ".html", ".htm", ".css",
    # Code (courant)
    ".py", ".ipynb", ".sh", ".bat", ".ps1",
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".java", ".kt", ".kts",
    ".go", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cc",
    ".cs",
    ".php", ".rb", ".swift", ".scala", ".lua",
    ".sql",
    # Bureautique modernes (extract texte best-effort côté indexeur/extracteurs)
    ".pdf", ".docx", ".pptx", ".xlsx",
}

def is_supported_ext(filename_or_ext: str) -> bool:
    """
    True si l'extension du fichier est dans SUPPORTED_EXTS.
    - Accepte un nom de fichier ou une extension (avec ou sans point).
    """
    if not filename_or_ext:
        return False
    s = filename_or_ext.strip().lower()
    ext = s
    if "." in s:
        # Si c'est un chemin/nom => on prend la vraie extension
        _, ext = os.path.splitext(s)
        ext = ext or s  # si pas d'extension, retombe sur s (cas ".md" passé directement)
    if not ext.startswith("."):
        ext = "." + ext
    return ext in SUPPORTED_EXTS

def normalize_ext_list(exts: Iterable[str]) -> List[str]:
    """
    Nettoie une liste d’extensions (ajoute le point, lowercase, supprime doublons/vides).
    """
    seen: Set[str] = set()
    out: List[str] = []
    for e in exts or []:
        s = (e or "").strip().lower()
        if not s:
            continue
        if not s.startswith("."):
            s = "." + s
        if len(s) > 16:  # garde-fou : évite saisies aberrantes
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _extract_office_xml_text(path: str, members: List[str]) -> str:
    parts: List[str] = []
    with zipfile.ZipFile(path) as zf:
        for member in members:
            try:
                data = zf.read(member)
            except KeyError:
                continue
            try:
                root = ET.fromstring(data)
            except ET.ParseError:
                continue
            texts = [t.strip() for t in root.itertext() if t and t.strip()]
            if texts:
                parts.append("\n".join(texts))
    return "\n\n".join(parts)

def _extract_xlsx_text(path: str) -> str:
    parts: List[str] = []
    with zipfile.ZipFile(path) as zf:
        shared_strings: List[str] = []
        try:
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            shared_strings = ["".join(t.strip() for t in si.itertext() if t and t.strip()) for si in root]
        except KeyError:
            shared_strings = []
        except ET.ParseError:
            shared_strings = []

        for name in zf.namelist():
            if not re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name):
                continue
            try:
                root = ET.fromstring(zf.read(name))
            except ET.ParseError:
                continue
            rows: List[str] = []
            for cell in root.iter():
                if not cell.tag.endswith("c"):
                    continue
                cell_type = cell.attrib.get("t")
                value = None
                for child in cell:
                    if child.tag.endswith("v") and child.text is not None:
                        value = child.text.strip()
                        break
                if not value:
                    continue
                if cell_type == "s":
                    try:
                        idx = int(value)
                        value = shared_strings[idx]
                    except Exception:
                        pass
                rows.append(value)
            if rows:
                parts.append("\n".join(rows))
    return "\n\n".join(parts)

def read_supported_text(path: str) -> str:
    """
    Lecture best-effort pour tous les formats non-PDF supportés.
    - Texte/code/config: lecture UTF-8 tolérante
    - .docx/.pptx/.xlsx: extraction XML depuis l'archive Office Open XML
    """
    ext = Path(path).suffix.lower()
    if ext == ".docx":
        return _extract_office_xml_text(path, ["word/document.xml"])
    if ext == ".pptx":
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        return _extract_office_xml_text(path, members)
    if ext == ".xlsx":
        return _extract_xlsx_text(path)
    return Path(path).read_text(encoding="utf-8", errors="ignore")
