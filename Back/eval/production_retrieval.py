"""Offline-only adapter around Auxilium's unchanged production retrieval code."""
from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import Any

from .gold import GoldCase

BACK_DIR = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def require_offline_production_models() -> None:
    """Fail before index creation; never permit a silent online model lookup."""
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("production_retrieval requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    required = {
        "models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2": "4328cf26390c98c5e3c738b4460a05b95f4911f5",
        "models--BAAI--bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    }
    for directory, snapshot in required.items():
        root = cache / directory
        ref = root / "refs" / "main"
        snapshot_dir = root / "snapshots" / snapshot
        if not ref.exists() or ref.read_text(encoding="utf-8").strip() != snapshot or not snapshot_dir.is_dir():
            raise RuntimeError(f"offline model cache incomplete: {root} (expected snapshot {snapshot})")


def _semantics(case: GoldCase) -> str:
    return case.query_semantics if case.query_semantics in {
        "fact_lookup", "current_state", "decision", "chronology", "comparison", "procedure",
    } else "general_document_question"


def _name(meta: dict[str, Any]) -> str:
    return str(meta.get("file") or Path(str(meta.get("path") or "")).name)


def _matches(text: str, terms: list[str]) -> bool:
    return all(term.casefold() in text.casefold() for term in terms)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def run(cases: list[GoldCase], *, k: int = 10) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require_offline_production_models()
    # Imports happen only after the strict offline/cache preflight.
    from api.index_singleton import idx
    from api.iterative_retrieval import run_iterative_evidence_retrieval
    from rag_core.constants import FINAL_K, MAX_CONTEXT_CHARS, EMBED_MODEL_NAME
    from rag_core.retrieval import clip_context_blocks, fuse_contiguous_passages

    if EMBED_MODEL_NAME != EMBEDDING_MODEL:
        raise RuntimeError(f"unexpected embedding model: {EMBED_MODEL_NAME}")
    if idx.cross_encoder is None or idx.rerank_model_used != RERANKER_MODEL:
        raise RuntimeError(f"primary reranker required, got {idx.rerank_model_used!r}")
    if idx.faiss_index is None:
        raise RuntimeError("production FAISS index is unavailable")

    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        row: dict[str, Any] = {"id": case.id, "category": case.category, "question": case.question,
                                "scope": case.execution_scope, "engine": "production_retrieval"}
        try:
            instrumented = idx.search_instrumented(
                case.question, retrieve_k=k, top_k_faiss=100, hybrid_alpha=0.8,
                use_rerank=True, variant="hybrid_current", include_trace=True,
            )
            trace = instrumented.trace.to_dict() if instrumented.trace else {}
            pool = []
            for candidate in trace.get("candidates", []):
                candidate_id = candidate["candidate_id"]
                if candidate_id in trace.get("fusion_candidate_ids", []):
                    pool.append((float(candidate.get("hybrid_score") or 0.0), idx._meta_with_text(candidate_id)))
            evidence, rounds, decision = run_iterative_evidence_retrieval(
                candidate_pool=pool or instrumented.items, initial_evidence=instrumented.items,
                corpus=idx.corpus, query=case.question, semantics=_semantics(case),
            )
            context = clip_context_blocks(fuse_contiguous_passages(evidence), max_chars=MAX_CONTEXT_CHARS, keep=FINAL_K)
            retrieved_docs = [_name(meta) for _, meta in instrumented.items]
            retrieved_keys = {value for _, meta in instrumented.items for value in (_name(meta), str(meta.get("document_id") or ""))}
            context_docs = [_name(meta) for _, meta in context]
            expected = set(case.expected_document_ids)
            relevant = expected if case.execution_scope == "indexed_local" else set()
            retrieved_sets = [{value for meta in [meta for _, meta in instrumented.items[:rank]] for value in (_name(meta), str(meta.get("document_id") or ""))} for rank in (1, 3, 5, 10)]
            context_text = "\n".join(str(meta.get("text") or "") for _, meta in context)
            required = [_matches(context_text, fact) for fact in case.required_facts]
            exact_signal = (
                float(all(entity.casefold() in context_text.casefold() for entity in case.exact_entities))
                if case.exact_entities else None
            )
            retry_occurred = len(rounds) > 1
            row.update({
                "status": "ok", "scored": bool(relevant), "document_hit": float(bool(retrieved_keys & relevant)) if relevant else None,
                "recall@1": float(bool(retrieved_sets[0] & relevant)) if relevant else None,
                "recall@3": float(bool(retrieved_sets[1] & relevant)) if relevant else None,
                "recall@5": float(bool(retrieved_sets[2] & relevant)) if relevant else None,
                "recall@10": float(bool(retrieved_sets[3] & relevant)) if relevant else None,
                "context_recall": sum(required) / len(required) if relevant and required else (1.0 if relevant else None),
                "exact_entity_signal": exact_signal,
                "exact_entity_violation": (1.0 - exact_signal) if exact_signal is not None else None,
                "retry_success": float(retry_occurred and all(required)) if case.expected_behavior.get("should_retry") else None,
                "retrieved_documents": retrieved_docs, "context_documents": context_docs,
                "context_chunk_uids": [uid for _, meta in context for uid in (meta.get("fused_chunk_uids") or [meta.get("chunk_uid")])],
                "required_facts_missing": [fact for fact, present in zip(case.required_facts, required) if not present],
                "rounds": rounds, "sufficiency": {"sufficient": decision.sufficient, "reason": decision.reason},
                "candidate_count_before_rerank": len(trace.get("mmr_selected_ids") or []),
                "final_block_count": len(context), "trace_errors": trace.get("errors", []),
            })
        except Exception as exc:
            row.update({"status": "error", "scored": False, "error": f"{type(exc).__name__}: {exc}"})
        row["latency_ms"] = (time.perf_counter() - started) * 1000.0
        rows.append(row)
    return rows, summarize(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("document_hit", "recall@1", "recall@3", "recall@5", "recall@10", "context_recall",
               "exact_entity_signal", "exact_entity_violation", "retry_success", "candidate_count_before_rerank", "final_block_count")
    def summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"case_count": len(items), "scored_case_count": sum(bool(item.get("scored")) for item in items)}
        for metric in metrics:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            result[metric] = sum(values) / len(values) if values else None
        latencies = [float(item["latency_ms"]) for item in items if item.get("latency_ms") is not None]
        result.update({"latency_mean_ms": statistics.mean(latencies) if latencies else None,
                       "latency_p50_ms": _percentile(latencies, .50), "latency_p95_ms": _percentile(latencies, .95)})
        return result
    categories = sorted({item["category"] for item in rows})
    return {"global": summary(rows), "by_category": {category: summary([item for item in rows if item["category"] == category]) for category in categories}}
