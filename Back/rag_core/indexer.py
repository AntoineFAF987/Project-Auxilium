# -*- coding: utf-8 -*-
"""
Indexer RAG : ajout d'un mode de réindexation incrémentale et robustesse index vide.
- build_or_update() : reconstruction complète
- rebuild_incremental() : ne traite que les fichiers ajoutés/modifiés, supprime les supprimés/dés-autorisés
- GESTION CAS VIDE : BM25 et FAISS sûrs quand il n'y a plus aucun document
"""

import os
import json
import uuid
from collections import Counter
from datetime import datetime, timezone
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
    EXACT_MATCH_BONUS, EXACT_MATCH_MIN_CHARS, MMR_LAMBDA,
    PRELIMINARY_POOL_MULTIPLIER,
    FINAL_K, MAX_CONTEXT_CHARS,
)
from .anchor_scope import select_anchor_scope, select_anchor_scope_adaptive
from .context_sufficiency import select_adaptive_context
from .utils import (
    ensure_dir, file_fingerprint, _is_under, _safe_walk, tokenize_for_bm25, SUPPORTED_EXTS, read_supported_text,
)
from .document_model import DOCUMENT_SCHEMA_VERSION
from .document_parsing import parse_document
from .structure_chunking import chunk_document

# ----- PDF utils -----
from ingest_pdfs import (
    read_pdf, split_text_fast,
    CHUNK_CHARS, CHUNK_OVERLAP, VERBOSE_PDF, MAX_PDF_SIZE_MB, HARD_LIMIT_PAGES, MAX_TEXT_CHARS_PER_PDF,
)

from .constants import ANSWER_MIN_CE  # utilisé indirectement par l'orchestration
from .contracts import (
    RETRIEVAL_VARIANTS,
    RetrievalCandidate,
    RetrievalResult,
    RetrievalTrace,
    RetrievalVariant,
)


def mmr_select(
    query_vec: np.ndarray,
    cand_vecs: np.ndarray,
    cand_ids: np.ndarray,
    k: int = 8,
    lambda_mult: float = MMR_LAMBDA,
) -> List[int]:
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
        _ = self.embed_model.encode(["warmup"], convert_to_tensor=True, normalize_embeddings=NORMALIZE_EMBED)

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
        self.manifest_path = os.path.join(self.index_dir, "manifest.json")
        self.legacy_corpus_path = self.corpus_path
        self.legacy_emb_path = self.emb_path
        self.legacy_faiss_path = self.faiss_path
        self.loaded_schema_version = 1
        self.index_manifest: Dict = {}

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
        if self.loaded_schema_version != DOCUMENT_SCHEMA_VERSION:
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
        self.corpus_path = self.legacy_corpus_path
        self.emb_path = self.legacy_emb_path
        self.faiss_path = self.legacy_faiss_path
        manifest = {}
        try:
            manifest = json.loads(Path(self.manifest_path).read_text(encoding="utf-8"))
            artifacts = manifest.get("artifacts") or {}
            if int(manifest.get("schema_version", 1)) >= 2 and artifacts:
                resolved = {
                    name: (Path(self.index_dir) / relative).resolve()
                    for name, relative in artifacts.items()
                }
                root = Path(self.index_dir).resolve()
                if not all(path == root or root in path.parents for path in resolved.values()):
                    raise ValueError("Index manifest references an artifact outside index_dir")
                self.corpus_path = str(resolved["corpus"])
                self.emb_path = str(resolved["embeddings"])
                self.faiss_path = str(resolved["faiss"])
        except Exception:
            manifest = {}
        ok = all(os.path.exists(p) for p in [self.corpus_path, self.emb_path, self.faiss_path])
        if not ok:
            return False
        self.loaded_schema_version = 1
        try:
            self.loaded_schema_version = int(manifest.get("schema_version", 1))
        except Exception:
            # A v1 index stays readable until an explicit reindex. It is never
            # incrementally mixed with v2 rows.
            pass
        self.index_manifest = manifest
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
        # Active and legacy artifacts are intentionally left untouched. A new
        # generation is activated only after all files have been validated.

    def _save_all(self, corpus_rows, embs, faiss_index):
        generations = Path(self.index_dir) / "generations"
        generations.mkdir(parents=True, exist_ok=True)
        generation_name = (
            f"v{DOCUMENT_SCHEMA_VERSION}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        staging = generations / f".{generation_name}.staging"
        final_generation = generations / generation_name
        staging.mkdir(parents=False, exist_ok=False)
        corpus_path = staging / "corpus.jsonl"
        emb_path = staging / "embeddings.npy"
        faiss_path = staging / "faiss.index"
        with corpus_path.open("w", encoding="utf-8") as f:
            for row in corpus_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        np.save(emb_path, embs.astype("float32"))
        faiss.write_index(faiss_index, str(faiss_path))
        self._validate_generation(corpus_rows, embs, faiss_index)
        persisted_embs = np.load(emb_path, mmap_mode="r")
        persisted_faiss = faiss.read_index(str(faiss_path))
        self._validate_persisted_generation(corpus_path, len(corpus_rows), persisted_embs, persisted_faiss)
        del persisted_embs
        os.replace(staging, final_generation)

        relative_root = final_generation.relative_to(Path(self.index_dir))
        summary = self._index_summary(corpus_rows)
        manifest = {
            "schema_version": DOCUMENT_SCHEMA_VERSION,
            "chunking": "structure-aware",
            "chunk_max_chars": CHUNK_CHARS,
            "oversized_unit_overlap": CHUNK_OVERLAP,
            "chunk_count": len(corpus_rows),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generation": generation_name,
            "artifacts": {
                "corpus": str(relative_root / "corpus.jsonl").replace("\\", "/"),
                "embeddings": str(relative_root / "embeddings.npy").replace("\\", "/"),
                "faiss": str(relative_root / "faiss.index").replace("\\", "/"),
            },
            "legacy_v1_retained": all(Path(path).exists() for path in (
                self.legacy_corpus_path, self.legacy_emb_path, self.legacy_faiss_path
            )),
            "summary": summary,
        }
        manifest_tmp = Path(self.index_dir) / f".manifest-{uuid.uuid4().hex}.tmp"
        manifest_tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(manifest_tmp, self.manifest_path)
        self.corpus_path = str(final_generation / "corpus.jsonl")
        self.emb_path = str(final_generation / "embeddings.npy")
        self.faiss_path = str(final_generation / "faiss.index")
        self.index_manifest = manifest
        self.loaded_schema_version = DOCUMENT_SCHEMA_VERSION
        print("💾 Index sauvegardé.")

    @staticmethod
    def _validate_generation(corpus_rows, embs, faiss_index) -> None:
        if len(corpus_rows) != int(embs.shape[0]) or len(corpus_rows) != int(faiss_index.ntotal):
            raise RuntimeError("Index generation count mismatch between corpus, embeddings and FAISS")
        required = {
            "schema_version", "document_id", "chunk_uid", "chunk_id", "order",
            "block_ids", "previous_chunk_uid", "next_chunk_uid", "source", "path", "text",
        }
        seen = set()
        by_document: Dict[str, List[Dict]] = {}
        for row in corpus_rows:
            missing = required - set(row)
            if missing:
                raise RuntimeError(f"Invalid v2 chunk metadata; missing {sorted(missing)}")
            if int(row["schema_version"]) != DOCUMENT_SCHEMA_VERSION:
                raise RuntimeError("Mixed index schemas are forbidden")
            uid = str(row["chunk_uid"])
            if uid in seen:
                raise RuntimeError(f"Duplicate chunk_uid: {uid}")
            seen.add(uid)
            by_document.setdefault(str(row["document_id"]), []).append(row)
        for rows in by_document.values():
            rows.sort(key=lambda item: int(item["order"]))
            for index, row in enumerate(rows):
                expected_previous = rows[index - 1]["chunk_uid"] if index else None
                expected_next = rows[index + 1]["chunk_uid"] if index + 1 < len(rows) else None
                if row.get("previous_chunk_uid") != expected_previous or row.get("next_chunk_uid") != expected_next:
                    raise RuntimeError(f"Broken chunk neighbor chain for {row['chunk_uid']}")

    @staticmethod
    def _validate_persisted_generation(corpus_path: Path, expected_count: int, embs, faiss_index) -> None:
        """Validate persisted JSONL without materialising the entire corpus again."""
        if expected_count != int(embs.shape[0]) or expected_count != int(faiss_index.ntotal):
            raise RuntimeError("Persisted index count mismatch between corpus, embeddings and FAISS")
        required = {
            "schema_version", "document_id", "chunk_uid", "chunk_id", "order",
            "block_ids", "previous_chunk_uid", "next_chunk_uid", "source", "path", "text",
        }
        seen: set[str] = set()
        count = 0
        with corpus_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = required - set(row)
                if missing:
                    raise RuntimeError(f"Invalid persisted v2 chunk metadata; missing {sorted(missing)}")
                if int(row["schema_version"]) != DOCUMENT_SCHEMA_VERSION:
                    raise RuntimeError("Mixed persisted index schemas are forbidden")
                uid = str(row["chunk_uid"])
                if uid in seen:
                    raise RuntimeError(f"Duplicate persisted chunk_uid: {uid}")
                seen.add(uid)
                count += 1
        if count != expected_count:
            raise RuntimeError("Persisted corpus row count mismatch")

    @staticmethod
    def _index_summary(corpus_rows) -> Dict:
        documents = {str(row.get("document_id")) for row in corpus_rows}
        sources = Counter(str(row.get("source") or "unknown") for row in corpus_rows)
        first_chunks = [row for row in corpus_rows if int(row.get("order", 0)) == 0]
        last_uids = {
            row.get("chunk_uid") for row in corpus_rows if row.get("next_chunk_uid") is None
        }
        return {
            "documents": len(documents),
            "chunks": len(corpus_rows),
            "chunks_by_source": dict(sorted(sources.items())),
            "chunks_without_section_id": sum(not row.get("section_id") for row in corpus_rows),
            "pdf_chunks_without_page": sum(
                row.get("source") == "pdf" and row.get("page") is None for row in corpus_rows
            ),
            "chunks_without_previous": sum(not row.get("previous_chunk_uid") for row in corpus_rows),
            "chunks_without_next": sum(not row.get("next_chunk_uid") for row in corpus_rows),
            "expected_first_chunks": len(first_chunks),
            "expected_last_chunks": len(last_uids),
        }

    def _rows_for_path(self, path: str) -> Tuple[List[Dict], List[str]]:
        document = parse_document(path)
        fp = file_fingerprint(path)
        rows, texts = [], []
        for chunk in chunk_document(document):
            row = {
                "schema_version": DOCUMENT_SCHEMA_VERSION,
                "file": document.file,
                "path": document.path,
                "fingerprint": fp,
                "source": document.source,
                "title": document.title,
                "document_id": document.document_id,
                "parent_document_id": document.parent_document_id,
                "relation_type": document.relation_type,
                "document_metadata": document.source_metadata,
                **chunk.to_dict(),
                "text_len": len(chunk.text),
            }
            rows.append(row)
            texts.append(chunk.text)
        return rows, texts

    # --- encode ---
    def _batched_encode(self, texts: List[str], bs: int = 64) -> np.ndarray:
        outs = []
        print("🧮 Étape 3: embeddings…")
        with torch.no_grad():
            from tqdm import trange as _tr
            for i in _tr(0, len(texts), bs, desc="🔢 Encodage"):
                batch = texts[i:i+bs]
                emb = self.embed_model.encode(batch, convert_to_tensor=True, normalize_embeddings=NORMALIZE_EMBED)
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

                rows, document_texts = self._rows_for_path(path)
                if not document_texts:
                    print(f"⚠️  Vide ou illisible: {path}")
                    continue
                corpus_rows.extend(rows)
                texts.extend(document_texts)
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

        if self.loaded_schema_version != DOCUMENT_SCHEMA_VERSION and self.roots:
            print(
                f"ℹ️  Index schema v{self.loaded_schema_version} détecté → "
                f"reconstruction propre en v{DOCUMENT_SCHEMA_VERSION}."
            )
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
                rows, texts = self._rows_for_path(pth)
            except Exception as e:
                print(f"❌ Erreur lecture fichier {pth}: {e}")
                return [], [], np.zeros((0, self.embeddings.shape[1]), dtype="float32")
            if not texts:
                print(f"⚠️  Vide ou illisible: {pth}")
                return [], [], np.zeros((0, self.embeddings.shape[1]), dtype="float32")
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

    def search(self, query: str, retrieve_k: int = RETRIEVE_K, top_k_faiss: int = TOP_K_FAISS, hybrid_alpha: float = HYBRID_ALPHA, use_rerank: bool = True, allowed_sources: Optional[set] = None, document_ids: Optional[set[str]] = None) -> Tuple[List[Tuple[float, Dict]], List[float]]:
        if self.faiss_index is None:
            raise RuntimeError("Index introuvable; lance build_or_update() d'abord.")
        if not self.metas:
            # Index vide : rien à renvoyer
            return [], []

        with torch.no_grad():
            q = self.embed_model.encode(query, convert_to_tensor=True, normalize_embeddings=NORMALIZE_EMBED)
        q_np = q.detach().cpu().numpy().astype("float32").reshape(1, -1)

        scoped_ids = {str(value) for value in (document_ids or set()) if value}
        scope_indices = np.array([i for i, meta in enumerate(self.metas) if not scoped_ids or str(meta.get("document_id") or "") in scoped_ids], dtype=int)
        if scoped_ids and scope_indices.size == 0:
            return [], []
        k_faiss = min(top_k_faiss, len(scope_indices))
        if k_faiss <= 0:
            return [], []

        if scoped_ids:
            local_scores = np.dot(self.embeddings[scope_indices], q_np[0])
            order = np.argsort(local_scores)[-k_faiss:][::-1]
            faiss_scores, faiss_ids = local_scores[order], scope_indices[order]
        else:
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

        top_bm25 = scope_indices[np.argsort(bm25_scores_full[scope_indices])[-TOP_K_BM25:]]
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
            bonus = EXACT_MATCH_BONUS if (len(q_lower) >= EXACT_MATCH_MIN_CHARS and q_lower in txt_lower) else 0.0
            score = hybrid_alpha * bm_score + (1 - hybrid_alpha) * vec_score + bonus
            fused.append((score, int(idx)))
        fused.sort(key=lambda x: x[0], reverse=True)

        prelim_k = max(retrieve_k * PRELIMINARY_POOL_MULTIPLIER, retrieve_k)
        cand_ids_sorted = np.array([idx for _, idx in fused[:prelim_k]], dtype=int)
        if cand_ids_sorted.size == 0:
            return [], []
        cand_vecs = self.embeddings[cand_ids_sorted]
        q_vec = q_np[0]
        mmr_ids = mmr_select(q_vec, cand_vecs, cand_ids_sorted, k=retrieve_k, lambda_mult=MMR_LAMBDA)
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

    # -------------------- OPT-IN INSTRUMENTATION --------------------
    def search_instrumented(
        self,
        query: str,
        retrieve_k: int = RETRIEVE_K,
        top_k_faiss: int = TOP_K_FAISS,
        hybrid_alpha: float = HYBRID_ALPHA,
        use_rerank: bool = True,
        allowed_sources: Optional[set] = None,
        *,
        variant: RetrievalVariant = "hybrid_current",
        include_trace: bool = True,
        document_ids: Optional[set[str]] = None,
    ) -> RetrievalResult:
        """Run observable retrieval without changing the production ``search``.

        ``variant='hybrid_current'`` intentionally mirrors ``search()`` line by
        line and is protected by parity tests. Other variants are benchmark-only
        feature flags; none is used by the application runtime.
        """

        if variant not in RETRIEVAL_VARIANTS:
            raise ValueError(f"Unknown retrieval variant: {variant}")
        if self.faiss_index is None:
            raise RuntimeError("Index introuvable; lance build_or_update() d'abord.")
        if not self.metas:
            trace = RetrievalTrace(
                query=query,
                variant=variant,
                parameters=self._trace_parameters(
                    retrieve_k, top_k_faiss, hybrid_alpha, use_rerank, allowed_sources
                ),
            ) if include_trace else None
            return RetrievalResult(items=[], ce_scores=[], trace=trace)

        with torch.no_grad():
            q = self.embed_model.encode(query, convert_to_tensor=True, normalize_embeddings=NORMALIZE_EMBED)
        q_np = q.detach().cpu().numpy().astype("float32").reshape(1, -1)

        scoped_ids = {str(value) for value in (document_ids or set()) if value}
        scope_indices = np.array([i for i, meta in enumerate(self.metas) if not scoped_ids or str(meta.get("document_id") or "") in scoped_ids], dtype=int)
        if scoped_ids and scope_indices.size == 0:
            return RetrievalResult(items=[], ce_scores=[], trace=None)
        k_faiss = min(top_k_faiss, len(scope_indices))
        if k_faiss <= 0:
            return RetrievalResult(items=[], ce_scores=[], trace=None)

        if scoped_ids:
            local_scores = np.dot(self.embeddings[scope_indices], q_np[0])
            order = np.argsort(local_scores)[-k_faiss:][::-1]
            faiss_scores, faiss_ids = local_scores[order], scope_indices[order]
        else:
            faiss_scores, faiss_ids = self.faiss_index.search(q_np, k_faiss)
            faiss_scores, faiss_ids = faiss_scores[0], faiss_ids[0]

        bm25_scores_full = []
        bm25_available = False
        if self.bm25_index is not None:
            try:
                bm25_scores_full = self.bm25_index.get_scores(tokenize_for_bm25(query))
                bm25_available = True
            except Exception:
                bm25_scores_full = []
        bm25_scores_full = np.array(bm25_scores_full, dtype=np.float32) if len(bm25_scores_full) else np.zeros(len(self.metas), dtype=np.float32)
        if len(bm25_scores_full) > 0:
            bmin, bmax = float(np.min(bm25_scores_full)), float(np.max(bm25_scores_full))
            bm25_scores_full = ((bm25_scores_full - bmin) / (bmax - bmin) if bmax > bmin else np.zeros_like(bm25_scores_full, dtype=np.float32))

        top_bm25 = scope_indices[np.argsort(bm25_scores_full[scope_indices])[-TOP_K_BM25:]]
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
                trace = RetrievalTrace(
                    query=query,
                    variant=variant,
                    parameters=self._trace_parameters(
                        retrieve_k, top_k_faiss, hybrid_alpha, use_rerank, allowed_sources
                    ),
                ) if include_trace else None
                return RetrievalResult(items=[], ce_scores=[], trace=trace)

        candidate_map: Dict[int, RetrievalCandidate] = {}
        union_ids = [int(i) for i in cand_ids_all]

        def candidate(idx_id: int) -> RetrievalCandidate:
            idx_int = int(idx_id)
            if idx_int not in candidate_map:
                metadata = dict(self.metas[idx_int])
                metadata["idx"] = idx_int
                candidate_map[idx_int] = RetrievalCandidate(
                    candidate_id=idx_int,
                    metadata=metadata,
                )
            return candidate_map[idx_int]

        union_id_set = set(union_ids)
        for rank, idx_id in enumerate(union_ids, start=1):
            candidate(idx_id).union_rank = rank

        dense_ids = [int(i) for i in faiss_ids if i != -1 and int(i) in union_id_set]
        for rank, idx_id in enumerate(dense_ids, start=1):
            item = candidate(idx_id)
            item.dense_score = faiss_map[idx_id]
            item.dense_rank = rank

        bm25_ids = sorted(
            (int(i) for i in top_bm25 if int(i) in union_id_set),
            key=lambda idx_id: float(bm25_scores_full[idx_id]),
            reverse=True,
        )
        for rank, idx_id in enumerate(bm25_ids, start=1):
            item = candidate(idx_id)
            if bm25_available:
                item.bm25_score = float(bm25_scores_full[idx_id])
            item.bm25_rank = rank

        parameters = self._trace_parameters(
            retrieve_k, top_k_faiss, hybrid_alpha, use_rerank, allowed_sources
        )

        # Benchmark-only single-signal baselines. They do not enter search().
        if variant == "dense_only":
            chosen_ids = dense_ids[:retrieve_k]
            items = [(float(faiss_map[idx_id]), self._meta_with_text(idx_id)) for idx_id in chosen_ids]
            trace = self._build_trace(
                query, variant, parameters, candidate_map, dense_ids, bm25_ids,
                union_ids, [], [], [], [], chosen_ids,
            ) if include_trace else None
            return RetrievalResult(items=items, ce_scores=[], trace=trace)

        if variant == "bm25_only":
            chosen_ids = bm25_ids[:retrieve_k]
            items = [(float(bm25_scores_full[idx_id]), self._meta_with_text(idx_id)) for idx_id in chosen_ids]
            trace = self._build_trace(
                query, variant, parameters, candidate_map, dense_ids, bm25_ids,
                union_ids, [], [], [], [], chosen_ids,
            ) if include_trace else None
            return RetrievalResult(items=items, ce_scores=[], trace=trace)

        fused = []
        q_lower = query.lower()
        for idx_id in cand_ids_all:
            idx_int = int(idx_id)
            vec_score = faiss_map.get(idx_int, 0.0)
            bm_score = float(bm25_scores_full[idx_int]) if 0 <= idx_int < len(bm25_scores_full) else 0.0
            txt_lower = self._get_text(idx_int).lower()
            bonus = EXACT_MATCH_BONUS if (len(q_lower) >= EXACT_MATCH_MIN_CHARS and q_lower in txt_lower) else 0.0
            score = hybrid_alpha * bm_score + (1 - hybrid_alpha) * vec_score + bonus
            item = candidate(idx_int)
            if idx_int in faiss_map:
                item.dense_score = vec_score
            if bm25_available:
                item.bm25_score = bm_score
            item.exact_match_bonus = bonus
            item.hybrid_score = float(score)
            fused.append((score, idx_int))
        fused.sort(key=lambda x: x[0], reverse=True)
        fusion_ids = [idx_id for _, idx_id in fused]
        for rank, idx_id in enumerate(fusion_ids, start=1):
            candidate(idx_id).fusion_rank = rank

        prelim_k = max(retrieve_k * PRELIMINARY_POOL_MULTIPLIER, retrieve_k)
        cand_ids_sorted = np.array([idx_id for _, idx_id in fused[:prelim_k]], dtype=int)
        top_pool_ids = [int(i) for i in cand_ids_sorted]
        for rank, idx_id in enumerate(top_pool_ids, start=1):
            candidate(idx_id).top_pool_rank = rank
        if cand_ids_sorted.size == 0:
            trace = self._build_trace(
                query, variant, parameters, candidate_map, dense_ids, bm25_ids,
                union_ids, fusion_ids, top_pool_ids, [], [], [],
            ) if include_trace else None
            return RetrievalResult(items=[], ce_scores=[], trace=trace)

        if variant == "hybrid_no_mmr":
            stage_ids = top_pool_ids[:retrieve_k]
            mmr_ids: List[int] = []
        else:
            cand_vecs = self.embeddings[cand_ids_sorted]
            q_vec = q_np[0]
            mmr_ids = mmr_select(q_vec, cand_vecs, cand_ids_sorted, k=retrieve_k, lambda_mult=MMR_LAMBDA)
            stage_ids = mmr_ids
            if include_trace:
                mmr_scores = (
                    self._mmr_scores_for_selected(
                        q_vec,
                        cand_vecs,
                        cand_ids_sorted,
                        mmr_ids,
                        lambda_mult=MMR_LAMBDA,
                    )
                    if cand_ids_sorted.size > retrieve_k
                    else {}
                )
                for rank, idx_id in enumerate(mmr_ids, start=1):
                    item = candidate(idx_id)
                    item.selected_by_mmr = True
                    item.mmr_rank = rank
                    item.mmr_score = mmr_scores.get(idx_id)

        prelim = [(0.0, idx_id) for idx_id in stage_ids]
        ce_scores_list: List[float] = []
        reranked_ids: List[int] = []
        rerank_enabled = use_rerank and variant != "hybrid_no_reranker"
        if rerank_enabled and self.cross_encoder is not None and len(prelim) > 0:
            pairs = [(query, self._get_text(idx_id)) for _, idx_id in prelim]
            try:
                ce_scores = self.cross_encoder.predict(pairs)
                ce_scores_list = [float(score) for score in ce_scores]
                for idx_id, score in zip(stage_ids, ce_scores_list):
                    candidate(idx_id).reranker_score = score
                reranked = sorted(
                    zip(ce_scores_list, stage_ids), key=lambda x: x[0], reverse=True
                )
                chosen = reranked[:retrieve_k]
                reranked_ids = [idx_id for _, idx_id in chosen]
                for rank, idx_id in enumerate(reranked_ids, start=1):
                    candidate(idx_id).reranker_rank = rank
                if variant in {"anchor_scope", "anchor_scope_adaptive", "anchor_scope_adaptive_k"}:
                    scope_selector = (
                        select_anchor_scope_adaptive
                        if variant in {"anchor_scope_adaptive", "anchor_scope_adaptive_k"}
                        else select_anchor_scope
                    )
                    scoped, anchor_scope_trace = scope_selector(
                        query=query,
                        query_vec=q_np[0],
                        metas=self.metas,
                        texts=self.texts,
                        embeddings=self.embeddings,
                        bm25_scores=bm25_scores_full,
                        baseline_ids=reranked_ids,
                        candidate_signals={
                            idx_id: item.to_dict() for idx_id, item in candidate_map.items()
                        },
                        cross_encoder=self.cross_encoder if use_rerank else None,
                        max_context_chars=MAX_CONTEXT_CHARS,
                        final_k=FINAL_K,
                    )
                    sufficiency_trace = None
                    if variant == "anchor_scope_adaptive_k":
                        protected_ids = (
                            list(anchor_scope_trace.get("selected_anchor_ids") or [])
                            + list(anchor_scope_trace.get("selected_scope_ids") or [])
                        )
                        scoped, sufficiency_trace = select_adaptive_context(
                            query=query,
                            candidates=scoped,
                            metas=self.metas,
                            texts=self.texts,
                            candidate_signals={
                                idx_id: item.to_dict() for idx_id, item in candidate_map.items()
                            },
                            protected_candidate_ids=protected_ids,
                            final_k=FINAL_K,
                        )
                    final_ids = [idx_id for _, idx_id in scoped]
                    for idx_id in final_ids:
                        candidate(idx_id)
                    items = [
                        (float(score), self._meta_with_text(idx_id))
                        for score, idx_id in scoped
                    ]
                    trace = self._build_trace(
                        query, variant, parameters, candidate_map, dense_ids, bm25_ids,
                        union_ids, fusion_ids, top_pool_ids, mmr_ids, reranked_ids, final_ids,
                    ) if include_trace else None
                    if trace is not None:
                        trace.anchor_scope = anchor_scope_trace
                        trace.sufficiency = sufficiency_trace
                    return RetrievalResult(items=items, ce_scores=ce_scores_list, trace=trace)
                items = [
                    (float(score), self._meta_with_text(idx_id))
                    for score, idx_id in chosen
                ]
                trace = self._build_trace(
                    query, variant, parameters, candidate_map, dense_ids, bm25_ids,
                    union_ids, fusion_ids, top_pool_ids, mmr_ids, reranked_ids, reranked_ids,
                ) if include_trace else None
                return RetrievalResult(items=items, ce_scores=ce_scores_list, trace=trace)
            except Exception as exc:
                trace_error = f"reranker: {type(exc).__name__}: {exc}"
        else:
            trace_error = None

        final_ids = stage_ids
        anchor_scope_trace = None
        if variant in {"anchor_scope", "anchor_scope_adaptive", "anchor_scope_adaptive_k"}:
            scope_selector = (
                select_anchor_scope_adaptive
                if variant in {"anchor_scope_adaptive", "anchor_scope_adaptive_k"}
                else select_anchor_scope
            )
            scoped, anchor_scope_trace = scope_selector(
                query=query,
                query_vec=q_np[0],
                metas=self.metas,
                texts=self.texts,
                embeddings=self.embeddings,
                bm25_scores=bm25_scores_full,
                baseline_ids=final_ids,
                candidate_signals={
                    idx_id: item.to_dict() for idx_id, item in candidate_map.items()
                },
                cross_encoder=None,
                max_context_chars=MAX_CONTEXT_CHARS,
                final_k=FINAL_K,
            )
            sufficiency_trace = None
            if variant == "anchor_scope_adaptive_k":
                protected_ids = (
                    list(anchor_scope_trace.get("selected_anchor_ids") or [])
                    + list(anchor_scope_trace.get("selected_scope_ids") or [])
                )
                scoped, sufficiency_trace = select_adaptive_context(
                    query=query,
                    candidates=scoped,
                    metas=self.metas,
                    texts=self.texts,
                    candidate_signals={
                        idx_id: item.to_dict() for idx_id, item in candidate_map.items()
                    },
                    protected_candidate_ids=protected_ids,
                    final_k=FINAL_K,
                )
            final_ids = [idx_id for _, idx_id in scoped]
            for idx_id in final_ids:
                candidate(idx_id)
            items = [(float(score), self._meta_with_text(idx_id)) for score, idx_id in scoped]
        else:
            items = [(0.0, self._meta_with_text(idx_id)) for idx_id in final_ids]
        trace = self._build_trace(
            query, variant, parameters, candidate_map, dense_ids, bm25_ids,
            union_ids, fusion_ids, top_pool_ids, mmr_ids, reranked_ids, final_ids,
        ) if include_trace else None
        if trace is not None and trace_error:
            trace.errors.append(trace_error)
        if trace is not None and anchor_scope_trace is not None:
            trace.anchor_scope = anchor_scope_trace
        if trace is not None and variant == "anchor_scope_adaptive_k":
            trace.sufficiency = sufficiency_trace
        return RetrievalResult(items=items, ce_scores=ce_scores_list, trace=trace)

    @staticmethod
    def _trace_parameters(
        retrieve_k: int,
        top_k_faiss: int,
        hybrid_alpha: float,
        use_rerank: bool,
        allowed_sources: Optional[set],
    ) -> Dict:
        return {
            "retrieve_k": int(retrieve_k),
            "top_k_faiss": int(top_k_faiss),
            "top_k_bm25": int(TOP_K_BM25),
            "hybrid_alpha": float(hybrid_alpha),
            "prelim_k": int(max(retrieve_k * PRELIMINARY_POOL_MULTIPLIER, retrieve_k)),
            "mmr_lambda": float(MMR_LAMBDA),
            "exact_match_bonus": float(EXACT_MATCH_BONUS),
            "exact_match_min_chars": int(EXACT_MATCH_MIN_CHARS),
            "use_rerank": bool(use_rerank),
            "allowed_sources": sorted(allowed_sources) if allowed_sources else None,
        }

    @staticmethod
    def _mmr_scores_for_selected(
        query_vec: np.ndarray,
        cand_vecs: np.ndarray,
        cand_ids: np.ndarray,
        selected_ids: List[int],
        lambda_mult: float,
    ) -> Dict[int, float]:
        q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        c = cand_vecs / (
            np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-9
        )
        id_to_pos = {int(idx_id): pos for pos, idx_id in enumerate(cand_ids)}
        selected_positions: List[int] = []
        scores: Dict[int, float] = {}
        sim_to_query = (c @ q).reshape(-1)
        for idx_id in selected_ids:
            pos = id_to_pos[int(idx_id)]
            if not selected_positions:
                score = float(sim_to_query[pos])
            else:
                max_sim_to_selected = float(
                    np.max(c[pos] @ c[selected_positions].T)
                )
                score = float(
                    lambda_mult * sim_to_query[pos]
                    - (1 - lambda_mult) * max_sim_to_selected
                )
            scores[int(idx_id)] = score
            selected_positions.append(pos)
        return scores

    @staticmethod
    def _build_trace(
        query: str,
        variant: RetrievalVariant,
        parameters: Dict,
        candidate_map: Dict[int, RetrievalCandidate],
        dense_ids: List[int],
        bm25_ids: List[int],
        union_ids: List[int],
        fusion_ids: List[int],
        top_pool_ids: List[int],
        mmr_ids: List[int],
        reranked_ids: List[int],
        final_ids: List[int],
    ) -> RetrievalTrace:
        for rank, idx_id in enumerate(final_ids, start=1):
            candidate_map[idx_id].final_rank = rank
        candidates = sorted(candidate_map.values(), key=lambda item: item.candidate_id)
        return RetrievalTrace(
            query=query,
            variant=variant,
            parameters=parameters,
            candidates=candidates,
            dense_candidate_ids=dense_ids,
            bm25_candidate_ids=bm25_ids,
            union_candidate_ids=union_ids,
            fusion_candidate_ids=fusion_ids,
            top_pool_candidate_ids=top_pool_ids,
            mmr_selected_ids=mmr_ids,
            reranked_candidate_ids=reranked_ids,
            final_candidate_ids=final_ids,
        )
