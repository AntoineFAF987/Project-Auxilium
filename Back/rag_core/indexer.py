# -*- coding: utf-8 -*-
"""
Indexer RAG : ajout d'un mode de réindexation incrémentale et robustesse index vide.
- build_or_update() : reconstruction complète
- rebuild_incremental() : ne traite que les fichiers ajoutés/modifiés, supprime les supprimés/dés-autorisés
- GESTION CAS VIDE : BM25 et FAISS sûrs quand il n'y a plus aucun document
"""

import os
import json
from typing import List, Dict, Tuple, Iterable, Optional
from pathlib import Path
import numpy as np
import torch
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from .constants import (
    EMBED_MODEL_NAME, RERANK_MODEL_NAME, DEVICE,
    TOP_K_FAISS, TOP_K_BM25, RETRIEVE_K, HYBRID_ALPHA, NORMALIZE_EMBED,
)
from .utils import (
    ensure_dir, file_fingerprint, _is_under, _safe_walk, tokenize_for_bm25, SUPPORTED_EXTS, read_supported_text,
)

# ----- PDF utils -----
from ingest_pdfs import (
    read_pdf, split_text_fast,
    CHUNK_CHARS, CHUNK_OVERLAP, VERBOSE_PDF, MAX_PDF_SIZE_MB, HARD_LIMIT_PAGES, MAX_TEXT_CHARS_PER_PDF,
)

from .constants import ANSWER_MIN_CE  # utilisé indirectement par l'orchestration


def mmr_select(query_vec: np.ndarray, cand_vecs: np.ndarray, cand_ids: np.ndarray, k: int = 8, lambda_mult: float = 0.7) -> List[int]:
    if cand_ids.size <= k:
        return list(map(int, cand_ids))
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    c = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-9)
    selected, candidates = [], list(range(len(cand_ids)))
    sim_to_query = (c @ q).reshape(-1)
    while candidates and len(selected) < k:
        if not selected:
            i = int(max(candidates, key=lambda idx: sim_to_query[idx])); selected.append(i); candidates.remove(i); continue
        max_sim_to_sel = np.max(c[candidates] @ c[selected].T, axis=1)
        mmr_scores = lambda_mult * sim_to_query[candidates] - (1 - lambda_mult) * max_sim_to_sel
        pick_pos = int(np.argmax(mmr_scores))
        i = candidates[pick_pos]
        selected.append(i); candidates.remove(i)
    return [int(cand_ids[i]) for i in selected]


class RAGIndexer:
    def __init__(self, roots: List[str], index_dir: str, exclude_globs: Optional[List[str]] = None, follow_symlinks: bool = False):
        self.roots = [os.path.abspath(r) for r in roots]
        self.index_dir = os.path.abspath(index_dir)
        self.exclude_globs = exclude_globs or []
        self.follow_symlinks = follow_symlinks
        ensure_dir(self.index_dir)

        print(f"➡️  Device: {DEVICE}")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
        _ = self.embed_model.encode(["warmup"], convert_to_tensor=True, normalize_embeddings=True)

        self.cross_encoder = None
        self.rerank_model_used = None
        try:
            # Essayer d'abord le reranker multilingue
            self.cross_encoder = CrossEncoder(RERANK_MODEL_NAME, device=DEVICE)
            self.rerank_model_used = RERANK_MODEL_NAME
            print(f"✅ Reranker multilingue chargé: {RERANK_MODEL_NAME}")
        except Exception as e:
            print(f"⚠️  Reranker multilingue indisponible ({e}), fallback sur modèle EN...")
            try:
                from .constants import RERANK_MODEL_FALLBACK
                self.cross_encoder = CrossEncoder(RERANK_MODEL_FALLBACK, device=DEVICE)
                self.rerank_model_used = RERANK_MODEL_FALLBACK
                print(f"✅ Reranker fallback chargé: {RERANK_MODEL_FALLBACK}")
            except Exception as e2:
                print(f"ℹ️  Aucun reranker disponible ({e2}). On continuera sans.")

        # Persistés
        self.corpus_path = os.path.join(self.index_dir, "corpus.jsonl")
        self.emb_path = os.path.join(self.index_dir, "embeddings.npy")
        self.faiss_path = os.path.join(self.index_dir, "faiss.index")

        # En mémoire
        self.corpus: List[Dict] = []
        self.metas: List[Dict] = []
        self.texts: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25_corpus_tokens: List[List[str]] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self.faiss_index = None

    # --- exclusions ---
    def _excluded(self, path: str) -> bool:
        if _is_under(path, self.index_dir):
            return True
        import fnmatch as _fnm
        norm = path.replace("\\", "/")
        for g in self.exclude_globs:
            if _fnm.fnmatch(norm, g):
                return True
        return False

    def _iter_docs(self) -> Iterable[str]:
        for root in self.roots:
            for cur_root, _, files in _safe_walk(root, followlinks=self.follow_symlinks):
                if self._excluded(cur_root):
                    continue
                for fn in files:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in SUPPORTED_EXTS:
                        p = os.path.join(cur_root, fn)
                        if not self._excluded(p):
                            yield p

    # --- changements ---
    def _current_doc_fingerprints(self) -> Dict[str, str]:
        fps = {}
        for p in self._iter_docs():
            try:
                fps[p] = file_fingerprint(p)
            except Exception:
                continue
        return fps

    def needs_rebuild(self) -> bool:
        # Cas "aucun root" et "index vide" => pas besoin de rebuild
        if not self.roots and not self.metas:
            return False
        if not self.metas:
            return True
        curr = self._current_doc_fingerprints()
        indexed = {m["path"]: m["fingerprint"] for m in self.metas}
        for p in curr:
            if p not in indexed: return True
        for p, fp in curr.items():
            if p in indexed and indexed[p] != fp: return True
        for p in indexed:
            if p not in curr: return True
        return False

    # --- persistance ---
    def _load_existing(self) -> bool:
        ok = all(os.path.exists(p) for p in [self.corpus_path, self.emb_path, self.faiss_path])
        if not ok:
            return False
        self.corpus = [json.loads(l) for l in open(self.corpus_path, "r", encoding="utf-8")]
        self.metas = [{k: v for k, v in c.items() if k != "text"} for c in self.corpus]
        self.texts = [c["text"] for c in self.corpus]
        self.embeddings = np.load(self.emb_path)
        self.faiss_index = faiss.read_index(self.faiss_path)
        self.bm25_corpus_tokens = [tokenize_for_bm25(t) for t in self.texts]
        self.bm25_index = BM25Okapi(self.bm25_corpus_tokens) if self.bm25_corpus_tokens else None
        print(f"✅ Index chargé ({len(self.metas)} chunks).")
        return True

    def _prepare_for_save(self):
        try:
            if isinstance(self.embeddings, np.memmap):
                del self.embeddings
            else:
                self.embeddings = None
        except Exception:
            pass
        import gc; gc.collect()
        for p in [self.corpus_path, self.emb_path, self.faiss_path]:
            try:
                if os.path.exists(p): os.remove(p)
            except PermissionError:
                try: os.replace(p, p + ".old")
                except Exception: pass

    def _save_all(self, corpus_rows, embs, faiss_index):
        with open(self.corpus_path, "w", encoding="utf-8") as f:
            for row in corpus_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        np.save(self.emb_path, embs.astype("float32"))
        faiss.write_index(faiss_index, self.faiss_path)
        print("💾 Index sauvegardé.")

    # --- encode ---
    def _batched_encode(self, texts: List[str], bs: int = 64) -> np.ndarray:
        outs = []
        print("🧮 Étape 3: embeddings…")
        with torch.no_grad():
            from tqdm import trange as _tr
            for i in _tr(0, len(texts), bs, desc="🔢 Encodage"):
                batch = texts[i:i+bs]
                emb = self.embed_model.encode(batch, convert_to_tensor=True, normalize_embeddings=True)
                outs.append(emb.detach().cpu())
        return torch.cat(outs, dim=0).numpy().astype("float32")

    # --- build complet ---
    def build_or_update(self):
        print("📄 Étape 1: liste des documents.")
        docs = list(self._iter_docs())
        if not docs:
            roots_txt = " | ".join(self.roots)
            raise RuntimeError(f"Aucun document (PDF/TXT/MD) trouvé sous {roots_txt}.")
        print(f"→ {len(docs)} doc(s) trouvés sous: {', '.join(self.roots)}")

        print("✂️  Étape 2: lecture + chunking.")
        corpus_rows, texts = [], []
        for pi, path in enumerate(docs, 1):
            try:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                ext = Path(path).suffix.lower()

                if MAX_PDF_SIZE_MB > 0 and ext == ".pdf" and size_mb > MAX_PDF_SIZE_MB:
                    print(f"⏭️  Skip (PDF trop gros {size_mb:.1f} Mo) : {path}")
                    continue

                if VERBOSE_PDF:
                    if ext == ".pdf":
                        print(f"[{pi}/{len(docs)}] Lecture PDF: {path} ({size_mb:.1f} Mo)")
                    else:
                        print(f"[{pi}/{len(docs)}] Lecture TXT/MD: {path} ({size_mb:.1f} Mo)")

                if ext == ".pdf":
                    raw = read_pdf(path)
                else:
                    raw = read_supported_text(path)

                if not raw or not raw.strip():
                    print(f"⚠️  Vide ou illisible: {path}")
                    continue

                if ext == ".pdf" and MAX_TEXT_CHARS_PER_PDF > 0 and len(raw) > MAX_TEXT_CHARS_PER_PDF:
                    print(f"⚠️  Texte volumineux ({len(raw):,}) → tronqué à {MAX_TEXT_CHARS_PER_PDF:,}")
                    raw = raw[:MAX_TEXT_CHARS_PER_PDF]

                chunks = split_text_fast(raw)
                src = "email" if (ext in {".txt", ".md"} and raw.lstrip().startswith("Subject:")) else ("pdf" if ext == ".pdf" else "file")
                fp = file_fingerprint(path)

                for i, ch in enumerate(chunks):
                    row = {
                        "file": os.path.basename(path),
                        "path": path,
                        "fingerprint": fp,
                        "chunk_id": i,
                        "text_len": len(ch),
                        "source": src,
                        "text": ch,
                    }
                    corpus_rows.append(row); texts.append(ch)
            except Exception as e:
                print(f"❌ Erreur fichier {path}: {e}")

        if not texts:
            raise RuntimeError("Aucun texte exploitable.")

        embs = self._batched_encode(texts, bs=64)

        print("📦 Étape 4: FAISS.")
        d = embs.shape[1]
        faiss_index = faiss.IndexFlatIP(d)
        faiss_index.add(embs)

        print("🔎 Étape 5: BM25.")
        tokens = [tokenize_for_bm25(t) for t in texts]
        bm25 = BM25Okapi(tokens)

        print("🧹 Préparation à l’écriture…")
        self._prepare_for_save()
        print("💾 Étape 6: persistance.")
        self._save_all(corpus_rows, embs, faiss_index)

        self.corpus = corpus_rows
        self.metas = [{k: v for k, v in r.items() if k != "text"} for r in corpus_rows]
        self.texts = texts
        self.embeddings = embs
        self.faiss_index = faiss_index
        self.bm25_corpus_tokens = tokens
        self.bm25_index = bm25
        print(f"✅ Index reconstruit: {len(self.metas)} chunks.")

    # --- build incrémental ---
    def rebuild_incremental(self):
        """
        Ne lit/encode que les fichiers ajoutés ou modifiés.
        Supprime ceux disparus ou dés-autorisés.
        Gère aussi le cas où il ne reste plus AUCUN document autorisé.
        """

        # 0) Si aucun index sur disque ET aucun root autorisé → créer un index VIDE propre
        if not self._load_existing():
            if not self.roots:
                print("ℹ️  Aucun index existant et aucun dossier autorisé → création d'un index vide.")
                # dimension propre au modèle
                try:
                    d = int(getattr(self.embed_model, "get_sentence_embedding_dimension")())
                except Exception:
                    # fallback raisonnable pour mpnet
                    d = 768
                embs_all = np.zeros((0, d), dtype="float32")
                self.corpus = []
                self.metas = []
                self.texts = []
                self.embeddings = embs_all
                self.faiss_index = faiss.IndexFlatIP(d)

                # BM25 vide (stub)
                class _EmptyBM25:
                    def get_scores(self, tokens):
                        return np.zeros(0, dtype=np.float32)
                self.bm25_corpus_tokens = []
                self.bm25_index = _EmptyBM25()

                self._prepare_for_save()
                self._save_all([], embs_all, self.faiss_index)
                print("✅ Index vidé (aucun dossier autorisé).")
                return
            else:
                print("ℹ️  Aucun index existant → construction complète.")
                return self.build_or_update()

        # 1) On part d'un index existant, on scanne l'état actuel
        print("🔍 Scan fichiers (fingerprints)…")
        curr = self._current_doc_fingerprints()
        indexed = {m["path"]: m["fingerprint"] for m in self.metas}

        added = [p for p in curr if p not in indexed]
        removed = [p for p in indexed if p not in curr]
        changed = [p for p in curr if (p in indexed and indexed[p] != curr[p])]
        unchanged = [p for p in curr if (p in indexed and indexed[p] == curr[p])]

        print(f"➡️  Ajoutés: {len(added)} | Modifiés: {len(changed)} | Supprimés: {len(removed)} | Inchangés: {len(unchanged)}")

        # 2) Cartographie path -> indices (ancien index)
        path2idxs: Dict[str, List[int]] = {}
        for i, meta in enumerate(self.metas):
            path2idxs.setdefault(meta["path"], []).append(i)

        new_corpus: List[Dict] = []
        new_texts: List[str] = []
        new_embs: List[np.ndarray] = []

        # 3) Réutiliser les fichiers inchangés (texte + embeddings)
        for p in unchanged:
            for i in path2idxs.get(p, []):
                row = dict(self.corpus[i])  # inclut "text"
                new_corpus.append(row)
                new_texts.append(row["text"])
                new_embs.append(self.embeddings[i:i+1])

        # 4) (Ré)ingérer les fichiers ajoutés + modifiés
        def _ingest_file(pth: str) -> Tuple[List[Dict], List[str], np.ndarray]:
            size_mb = os.path.getsize(pth) / (1024 * 1024)
            ext = Path(pth).suffix.lower()
            if MAX_PDF_SIZE_MB > 0 and ext == ".pdf" and size_mb > MAX_PDF_SIZE_MB:
                print(f"⏭️  Skip (PDF trop gros {size_mb:.1f} Mo) : {pth}")
                return [], [], np.zeros((0, self.embeddings.shape[1]), dtype="float32")
            if VERBOSE_PDF:
                print(f"↻ Lecture {'PDF' if ext=='.pdf' else 'TXT/MD'}: {pth} ({size_mb:.1f} Mo)")
            try:
                raw = read_pdf(pth) if ext == ".pdf" else read_supported_text(pth)
            except Exception as e:
                print(f"❌ Erreur lecture fichier {pth}: {e}")
                return [], [], np.zeros((0, self.embeddings.shape[1]), dtype="float32")
            if not raw or not raw.strip():
                print(f"⚠️  Vide ou illisible: {pth}")
                return [], [], np.zeros((0, self.embeddings.shape[1]), dtype="float32")
            if ext == ".pdf" and MAX_TEXT_CHARS_PER_PDF > 0 and len(raw) > MAX_TEXT_CHARS_PER_PDF:
                print(f"⚠️  Texte volumineux ({len(raw):,}) → tronqué à {MAX_TEXT_CHARS_PER_PDF:,}")
                raw = raw[:MAX_TEXT_CHARS_PER_PDF]
            chunks = split_text_fast(raw)
            src = "email" if (ext in {".txt", ".md"} and raw.lstrip().startswith("Subject:")) else ("pdf" if ext == ".pdf" else "file")
            fp = file_fingerprint(pth)
            rows, texts = [], []
            for i, ch in enumerate(chunks):
                rows.append({
                    "file": os.path.basename(pth),
                    "path": pth,
                    "fingerprint": fp,
                    "chunk_id": i,
                    "text_len": len(ch),
                    "source": src,
                    "text": ch,
                })
                texts.append(ch)
            embs = self._batched_encode(texts, bs=64) if texts else np.zeros((0, self.embeddings.shape[1]), dtype="float32")
            return rows, texts, embs

        for p in added + changed:
            rows, texts, embs = _ingest_file(p)
            new_corpus.extend(rows)
            new_texts.extend(texts)
            if embs.size:
                new_embs.append(embs)

        # 5) Sauvegarde (FAISS/BM25) — y compris cas VIDE
        if new_embs:
            embs_all = np.concatenate(new_embs, axis=0).astype("float32")
        else:
            # Aucun texte exploitable (purge totale, par ex.)
            # On garde la même dimension d'embedding que l'ancien index
            d = int(self.embeddings.shape[1]) if (self.embeddings is not None and self.embeddings.ndim == 2) else \
                int(getattr(self.embed_model, "get_sentence_embedding_dimension", lambda: 768)())
            embs_all = np.zeros((0, d), dtype="float32")

        print("📦 FAISS (recréé à partir des embeddings consolidés)…")
        d = embs_all.shape[1]
        faiss_index = faiss.IndexFlatIP(d)
        if embs_all.size:
            faiss_index.add(embs_all)

        print("🔎 BM25 (recréé)…")
        tokens = [tokenize_for_bm25(t) for t in new_texts]
        if len(tokens) > 0:
            bm25 = BM25Okapi(tokens)
        else:
            # Stub BM25 pour éviter ZeroDivisionError quand pas de tokens/documents
            class _EmptyBM25:
                def get_scores(self, tokens_):
                    return np.zeros(0, dtype=np.float32)
            bm25 = _EmptyBM25()

        print("🧹 Préparation à l’écriture…")
        self._prepare_for_save()
        print("💾 Persistance (incrémentale)…")
        self._save_all(new_corpus, embs_all, faiss_index)

        # MAJ mémoire
        self.corpus = new_corpus
        self.metas = [{k: v for k, v in r.items() if k != "text"} for r in new_corpus]
        self.texts = new_texts
        self.embeddings = embs_all
        self.faiss_index = faiss_index
        self.bm25_corpus_tokens = tokens
        self.bm25_index = bm25

        print(f"✅ Index mis à jour (incrémental) : {len(self.metas)} chunks.")

    # -------------------- SEARCH --------------------
    def _get_text(self, idx: int) -> str:
        return self.corpus[idx]["text"]

    def _meta_with_text(self, idx: int) -> Dict:
        m = dict(self.metas[idx]); m["text"] = self._get_text(idx); m["idx"] = idx
        return m

    def _is_web_doc(self, meta: Dict) -> bool:
        try:
            if meta.get("url"): return True
            p = (meta.get("path") or "").lower()
            tokens = ("/web/", "\\web\\", "/crawl/", "\\crawl\\", "/internet/", "\\internet\\")
            return any(t in p for t in tokens)
        except Exception:
            return False

    def search(self, query: str, retrieve_k: int = RETRIEVE_K, top_k_faiss: int = TOP_K_FAISS, hybrid_alpha: float = HYBRID_ALPHA, use_rerank: bool = True, allowed_sources: Optional[set] = None) -> Tuple[List[Tuple[float, Dict]], List[float]]:
        if self.faiss_index is None:
            raise RuntimeError("Index introuvable; lance build_or_update() d'abord.")
        if not self.metas:
            # Index vide : rien à renvoyer
            return [], []

        with torch.no_grad():
            q = self.embed_model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        q_np = q.detach().cpu().numpy().astype("float32").reshape(1, -1)

        k_faiss = min(top_k_faiss, len(self.metas))
        if k_faiss <= 0:
            return [], []

        faiss_scores, faiss_ids = self.faiss_index.search(q_np, k_faiss)
        faiss_scores, faiss_ids = faiss_scores[0], faiss_ids[0]

        bm25_scores_full = []
        if self.bm25_index is not None:
            try:
                bm25_scores_full = self.bm25_index.get_scores(tokenize_for_bm25(query))
            except Exception:
                bm25_scores_full = []
        bm25_scores_full = np.array(bm25_scores_full, dtype=np.float32) if len(bm25_scores_full) else np.zeros(len(self.metas), dtype=np.float32)
        if len(bm25_scores_full) > 0:
            bmin, bmax = float(np.min(bm25_scores_full)), float(np.max(bm25_scores_full))
            bm25_scores_full = ((bm25_scores_full - bmin) / (bmax - bmin) if bmax > bmin else np.zeros_like(bm25_scores_full, dtype=np.float32))

        top_bm25 = np.argsort(bm25_scores_full)[-TOP_K_BM25:]
        faiss_map = {int(i): float(s) for s, i in zip(faiss_scores, faiss_ids) if i != -1}
        cand_ids_all = np.unique(np.concatenate([top_bm25.astype(int), faiss_ids[faiss_ids != -1].astype(int)]))

        if allowed_sources:
            kept = []
            for idx_id in cand_ids_all:
                meta = self.metas[int(idx_id)]
                src = (meta.get("source") or "").lower()
                is_web = self._is_web_doc(meta)
                if "web" in allowed_sources:
                    if is_web: kept.append(int(idx_id))
                else:
                    if (src in allowed_sources) and (not is_web):
                        kept.append(int(idx_id))
            cand_ids_all = np.array(kept, dtype=int)
            if cand_ids_all.size == 0:
                return [], []

        fused = []
        q_lower = query.lower()
        for idx in cand_ids_all:
            vec_score = faiss_map.get(int(idx), 0.0)
            bm_score = float(bm25_scores_full[int(idx)]) if 0 <= int(idx) < len(bm25_scores_full) else 0.0
            txt_lower = self._get_text(int(idx)).lower()
            bonus = 0.15 if (len(q_lower) >= 6 and q_lower in txt_lower) else 0.0
            score = hybrid_alpha * bm_score + (1 - hybrid_alpha) * vec_score + bonus
            fused.append((score, int(idx)))
        fused.sort(key=lambda x: x[0], reverse=True)

        prelim_k = max(retrieve_k * 2, retrieve_k)
        cand_ids_sorted = np.array([idx for _, idx in fused[:prelim_k]], dtype=int)
        if cand_ids_sorted.size == 0:
            return [], []
        cand_vecs = self.embeddings[cand_ids_sorted]
        q_vec = q_np[0]
        mmr_ids = mmr_select(q_vec, cand_vecs, cand_ids_sorted, k=retrieve_k, lambda_mult=0.7)
        prelim = [(0.0, i) for i in mmr_ids]

        ce_scores_list: List[float] = []
        if use_rerank and self.cross_encoder is not None and len(prelim) > 0:
            pairs = [(query, self._get_text(i)) for _, i in prelim]
            try:
                ce_scores = self.cross_encoder.predict(pairs)
                ce_scores_list = [float(s) for s in ce_scores]
                reranked = sorted(zip(ce_scores_list, [i for _, i in prelim]), key=lambda x: x[0], reverse=True)
                chosen = reranked[:retrieve_k]
                return [(float(s), self._meta_with_text(i)) for s, i in chosen], ce_scores_list
            except Exception:
                pass

        return [(0.0, self._meta_with_text(i)) for _, i in prelim], ce_scores_list
