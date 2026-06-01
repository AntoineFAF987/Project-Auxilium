# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 19:35:22 2025

Auteur : aejau (modifié : fallback web_live si pas de contexte local pertinent)
"""

# main.py
# -------------------------------------------------------------------
# Lanceur principal :
# - charge la config (en préservant toutes les clés, ex. graph / email_ingest)
# - exécute l’ingestion des emails (si config présente) AVANT l’index
# - gère l’index RAG (construction / MAJ)
# - boucle interactive (recherche + génération LLM)
# - commandes: showcfg, setroots, setindex, reindex, ingestmail, ingestpdfs, exit
# - nouveauté : si pas de contexte local pertinent → fallback web_live (Tavily)
# -------------------------------------------------------------------

import os, json
from pathlib import Path

# --- Modules internes
from rag_core import (
    RAGIndexer, ensure_dir, trim_history,
    keyword_overlap_count, format_context_for_llm,
    answerability_guard, ask_mistral_with_context, fuse_contiguous_passages,
    clip_context_blocks, RETRIEVE_K, TOP_K_FAISS, HYBRID_ALPHA,
    FUSE_ADJACENT_GAP, MAX_CONTEXT_CHARS, FINAL_K, HISTORY_MAX_TURNS,
    web_search_context  # ✅ web live (Tavily)
)

# Import optionnels : on veut que main.py tourne même si ces fichiers n'existent pas encore
try:
    from ingest_emails import ingest_emails
except Exception:
    ingest_emails = None

try:
    from ingest_pdfs import ingest_pdfs
except Exception:
    ingest_pdfs = None

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
DATA_DIR = APP_DIR / "data"

# -------------------------------------------------------------------
# Utilitaires CONFIG
# -------------------------------------------------------------------
def load_config_raw() -> dict:
    """
    Charge le JSON brut (toutes les clés) sans lever d'exception si vide/invalide.
    """
    cfg = {}
    if CONFIG_PATH.exists():
        raw = CONFIG_PATH.read_text(encoding="utf-8").strip()
        if not raw:
            print("⚠️  config.json est vide → on utilisera des défauts.")
            return {}
        try:
            cfg = json.loads(raw)
        except json.JSONDecodeError:
            print("⚠️  config.json invalide → on ignorera les champs invalides.")
            cfg = {}
    return cfg


def write_default_config_if_missing():
    """
    Crée un config.json minimal si absent.
    (On ne force pas les sections graph/email_ingest ici pour ne pas deviner tes IDs)
    """
    if CONFIG_PATH.exists():
        return
    default = {
        "roots": [str(APP_DIR.parent)],
        "index_dir": str(DATA_DIR / "index_rag"),
        "exclude_globs": [
            "*/Windows/*", "*/Program Files/*", "*/Program Files (x86)/*",
            "*/AppData/*", "*/$Recycle.Bin/*", "*/System Volume Information/*"
        ],
        "follow_symlinks": False
    }
    CONFIG_PATH.write_text(json.dumps(default, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ℹ️  config.json créé: {CONFIG_PATH}")


def config_get_core_settings(cfg: dict):
    """
    Extrait les paramètres cœur attendus par l'indexeur,
    en appliquant des valeurs par défaut si manquants.
    """
    roots = cfg.get("roots") or [str(APP_DIR.parent)]
    index_dir = cfg.get("index_dir") or str(DATA_DIR / "index_rag")
    exclude_globs = cfg.get("exclude_globs") or [
        "*/Windows/*", "*/Program Files/*", "*/Program Files (x86)/*",
        "*/AppData/*", "*/$Recycle.Bin/*", "*/System Volume Information/*"
    ]
    follow_symlinks = bool(cfg.get("follow_symlinks", False))
    return roots, index_dir, exclude_globs, follow_symlinks


def save_config_preserve(existing_cfg: dict, **patch):
    """
    Sauvegarde la config en PRÉSERVANT toutes les autres clés (graph, email_ingest, etc.).
    """
    new_cfg = dict(existing_cfg) if existing_cfg else {}
    new_cfg.update({k: v for k, v in patch.items() if v is not None})
    CONFIG_PATH.write_text(json.dumps(new_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return new_cfg


# -------------------------------------------------------------------
# Pré-ingestion EMAILS / PDFs (optionnelles)
# -------------------------------------------------------------------
def maybe_ingest_emails(cfg: dict):
    if ingest_emails is None:
        print("ℹ️  Module ingest_emails non trouvé (pas d’ingestion emails).")
        return
    if not cfg.get("graph") or not cfg.get("email_ingest"):
        print("ℹ️  Pas de configuration 'graph'/'email_ingest' → on saute l’ingestion emails.")
        return
    try:
        print("📧 Ingestion emails… (Microsoft Graph)")
        ingest_emails(str(CONFIG_PATH))
        print("✅ Ingestion emails terminée.")
    except Exception as ex:
        print(f"⚠️  Ingestion emails: {ex}")


def maybe_ingest_pdfs(cfg: dict):
    if ingest_pdfs is None:
        return
    try:
        print("📄 Ingestion PDFs…")
        ingest_pdfs(str(CONFIG_PATH))
        print("✅ Ingestion PDFs terminée.")
    except Exception as ex:
        print(f"⚠️  Ingestion PDFs: {ex}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
write_default_config_if_missing()

if __name__ == "__main__":
    # 1) Charge la config brute (toutes les clés)
    cfg = load_config_raw()

    # 2) Overrides via variables d'env si présents
    roots_env = os.getenv("RAG_ROOTS")  # ex: "C:\\;D:\\Docs"
    if roots_env:
        cfg["roots"] = [p.strip() for p in roots_env.split(";") if p.strip()]
    index_env = os.getenv("RAG_INDEX_DIR")
    if index_env:
        cfg["index_dir"] = index_env

    # 3) Extrait les réglages cœur pour l’indexeur
    roots, index_dir, exclude_globs, follow_symlinks = config_get_core_settings(cfg)

    # 4) S’assure que le dossier d’index existe
    ensure_dir(index_dir)

    # 5) (Option) Ingestion EMAILS avant la (re)construction d’index
    maybe_ingest_emails(cfg)

    # 6) (Option) Ingestion PDFs (décommente si tu veux la faire ici)
    # maybe_ingest_pdfs(cfg)

    print(f"📚 ROOTS: {roots}")
    print(f"💾 INDEX_DIR: {index_dir}")

    # 7) Indexation RAG
    idx = RAGIndexer(
        roots=roots,
        index_dir=index_dir,
        exclude_globs=exclude_globs,
        follow_symlinks=follow_symlinks
    )

    # Chargement / (re)construction
    if not Path(idx.corpus_path).exists():
        print("⚙️  Construction de l’index…")
        idx.build_or_update()
    else:
        if idx._load_existing():
            if idx.needs_rebuild():
                print("🔁 Changements détectés → reconstruction…")
                idx.build_or_update()
            else:
                print("✅ Index à jour.")
        else:
            print("⚙️  Index incomplet → reconstruction…")
            idx.build_or_update()

    # 8) Boucle interactive
    print("\nTape une question, ou commandes: 'reindex', 'ingestmail', 'ingestpdfs', 'setroots', 'setindex', 'showcfg', 'exit'")
    history_messages = []

    while True:
        q = input("> ").strip()
        if not q:
            continue
        cmd = q.lower()

        # --- commandes système
        if cmd == "exit":
            break

        if cmd == "showcfg":
            print(json.dumps(cfg, ensure_ascii=False, indent=2))
            continue

        if cmd == "setroots":
            print("Nouveaux roots ? (séparés par ;)  ex: C:\\;D:\\Docs;C:\\Users\\X\\Desktop")
            new = input("roots= ").strip()
            if new:
                new_roots = [p.strip() for p in new.split(";") if p.strip()]
                cfg = save_config_preserve(cfg, roots=new_roots)
                roots, index_dir, exclude_globs, follow_symlinks = config_get_core_settings(cfg)
                idx = RAGIndexer(roots, index_dir, exclude_globs, follow_symlinks)
                print("⚙️  Reconstruction…")
                idx.build_or_update()
            continue

        if cmd == "setindex":
            print("Nouveau index_dir ?  ex: C:\\IndexRAG")
            new = input("index_dir= ").strip()
            if new:
                ensure_dir(new)
                cfg = save_config_preserve(cfg, index_dir=new)
                roots, index_dir, exclude_globs, follow_symlinks = config_get_core_settings(cfg)
                idx = RAGIndexer(roots, index_dir, exclude_globs, follow_symlinks)
                print("⚙️  Reconstruction…")
                idx.build_or_update()
            continue

        if cmd == "reindex":
            print("⚙️  Reconstruction complète…")
            idx.build_or_update()
            continue

        if cmd == "ingestmail":
            maybe_ingest_emails(cfg)
            print("🔁 Reconstruction index après ingestion emails…")
            idx.build_or_update()
            continue

        if cmd == "ingestpdfs":
            maybe_ingest_pdfs(cfg)
            print("🔁 Reconstruction index après ingestion PDFs…")
            idx.build_or_update()
            continue

        # --- Recherche + génération LLM (avec fallback web_live)
        try:
            # Recherche locale (PDF/e-mails aplatis)
            prelim, ce_scores = idx.search(
                q,
                retrieve_k=RETRIEVE_K,
                top_k_faiss=TOP_K_FAISS,
                hybrid_alpha=HYBRID_ALPHA,
                use_rerank=True
            )

            # Fusion de passages adjacents + clipping de taille
            fused = fuse_contiguous_passages(prelim, gap=FUSE_ADJACENT_GAP)
            final_blocks = clip_context_blocks(fused, max_chars=MAX_CONTEXT_CHARS, keep=FINAL_K)

            gated_ok = answerability_guard(ce_scores, threshold=-0.20)
            pdf_context = format_context_for_llm(final_blocks) if final_blocks else ""

            # STRICT UNIQUEMENT si on a du contexte local pertinent
            overlap = keyword_overlap_count(q, pdf_context)
            use_strict = bool(pdf_context) and (gated_ok is True) and (overlap >= 2)

            context_for_llm = ""
            mode = "GENERAL(no-context)"

            if use_strict:
                context_for_llm = pdf_context
                mode = "STRICT(local)"
            else:
                # 🔁 Fallback web_live (Tavily) : on tente la recherche web
                web_txt = web_search_context(q, max_chars=4000, k=5) or ""
                if web_txt.strip():
                    context_for_llm = web_txt
                    mode = "STRICT(web_live)"
                else:
                    # on reste en général si rien trouvé sur le web
                    mode = "GENERAL(no-context)"

            print(f"[debug] mode={mode} | ctx_len={len(context_for_llm)} | local_overlap={overlap} | gated_ok={gated_ok}")

            # Historique court (pour le ton/fil, sans dépasser)
            temp_history = trim_history(history_messages + [{"role": "user", "content": q}], HISTORY_MAX_TURNS)

            # Appel LLM
            answer = ask_mistral_with_context(q, context_for_llm, temp_history)
            print("\n🤖 Réponse :", answer)

            # MàJ historique
            history_messages = temp_history + [{"role": "assistant", "content": answer}]

        except Exception as e:
            print(f"❌ Erreur: {e}")
