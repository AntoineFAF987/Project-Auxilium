from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .dataset import load_jsonl, validate_relevant_chunk_ids
from .metrics_retrieval import DEFAULT_KS, average_metrics, compute_query_metrics
from .report import aggregate_by_question_type, write_report
from .schemas import BenchmarkReport, QueryEvaluation, RetrievedChunk
from runtime_settings import RuntimeSettings, get_runtime_settings


BACK_DIR = Path(__file__).resolve().parent.parent


def effective_runtime_configuration(
    settings: RuntimeSettings,
    *,
    device: str,
) -> Dict[str, Any]:
    """Configuration canonique enregistrée par tous les benchmarks."""

    return {
        "configuration_hash": settings.config_hash(),
        "runtime": settings.effective_dict(),
        "resolved_device": device,
    }


def _load_config(path: Path) -> Dict[str, Any]:
    """Load the JSON-compatible YAML config without adding a YAML dependency."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"{path} must remain JSON-compatible YAML for the benchmark runner: {exc}"
        ) from exc
    if payload.get("pipeline") != "current":
        raise ValueError("This runner only supports pipeline='current'")
    return payload


def _resolve_from_back(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BACK_DIR / path


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=BACK_DIR.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_uid(meta: Dict[str, Any]) -> str:
    native = meta.get("chunk_uid")
    if native:
        return str(native)
    filename = str(meta.get("file") or Path(str(meta.get("path") or "unknown")).name)
    return f"{filename}::{meta.get('chunk_id', 'unknown')}"


def _document_uid(meta: Dict[str, Any]) -> str:
    filename = str(meta.get("file") or Path(str(meta.get("path") or "unknown")).name)
    return f"document::{filename}"


def _document_gold_ids(row) -> List[str]:
    documents = list(row.relevant_documents)
    if row.relevant_document:
        documents.append(row.relevant_document)
    documents.extend(span.document for span in row.evidence_spans if span.document)
    return [f"document::{value}" for value in dict.fromkeys(documents)]


def _resolved_locator_gold_ids(row, metas: List[Dict[str, Any]]) -> List[str]:
    """Resolve stable page/section/text locators against the current chunking."""
    documents = {
        value for value in [row.relevant_document, *row.relevant_documents] if value
    }
    matched: List[str] = []
    for meta in metas:
        filename = str(meta.get("file") or Path(str(meta.get("path") or "")).name)
        if documents and filename not in documents:
            continue
        page_match = not row.relevant_pages or meta.get("page") in row.relevant_pages
        section_match = not row.relevant_sections or meta.get("section") in row.relevant_sections
        span_match = not row.evidence_spans
        for span in row.evidence_spans:
            if span.document and span.document != filename:
                continue
            if span.page is not None and span.page != meta.get("page"):
                continue
            if span.section and span.section not in {
                meta.get("section"), *(meta.get("heading_path") or [])
            }:
                continue
            if span.text and span.text.casefold() not in str(meta.get("text") or "").casefold():
                continue
            span_match = True
            break
        if page_match and section_match and span_match:
            matched.append(_chunk_uid(meta))
    return list(dict.fromkeys(matched))


def is_premature_stop(
    *, selected_ids: List[str], available_ids: List[str], gold_ids: List[str]
) -> tuple[bool, bool]:
    """Return ``(premature, eligible)`` under the v2 documentary gold rule."""
    available_gold = set(available_ids) & set(gold_ids)
    eligible = bool(available_gold)
    return bool(eligible and not available_gold.issubset(set(selected_ids))), eligible


def run(
    config_path: Path,
    output_override: Path | None = None,
    *,
    variant_override: str | None = None,
    record_trace_override: bool | None = None,
) -> Path:
    # Existing runtime logs contain Unicode symbols. Keep the benchmark usable
    # from legacy Windows shells without changing the runtime logging code.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    config = _load_config(config_path)
    variant = variant_override or str(
        config.get("retrieval_variant", "hybrid_current")
    )
    record_trace = (
        bool(config.get("record_trace", False))
        if record_trace_override is None
        else record_trace_override
    )
    dataset_path = _resolve_from_back(str(config["dataset"])).resolve()
    rows = load_jsonl(dataset_path)
    ks = tuple(int(k) for k in config.get("metric_ks", DEFAULT_KS))

    # Importing the existing singleton is intentional: this is the exact
    # production retrieval object and production search method, unchanged.
    from api.index_singleton import idx
    from rag_core.constants import DEVICE
    from rag_core.contracts import RETRIEVAL_VARIANTS
    from rag_core.constants import FINAL_K, MAX_CONTEXT_CHARS
    from rag_core.retrieval import clip_context_blocks, fuse_contiguous_passages

    runtime_settings = get_runtime_settings()
    retrieval_settings = runtime_settings.retrieval

    if variant not in RETRIEVAL_VARIANTS:
        choices = ", ".join(sorted(RETRIEVAL_VARIANTS))
        raise ValueError(f"Unknown retrieval variant {variant!r}; choose one of: {choices}")

    available_chunk_ids = {_chunk_uid(meta) for meta in idx.metas}
    validate_relevant_chunk_ids(rows, available_chunk_ids)

    results: List[QueryEvaluation] = []
    for row in rows:
        started = time.perf_counter()
        retrieved_chunks: List[RetrievedChunk] = []
        retrieved_metas: List[Dict[str, Any]] = []
        retrieval_trace = None
        error = None
        use_chunk_gold = bool(row.relevant_chunk_ids) and all(
            value in available_chunk_ids for value in row.relevant_chunk_ids
        )
        resolved_locator_ids = _resolved_locator_gold_ids(row, idx.corpus) if (
            row.evidence_spans or row.relevant_pages or row.relevant_sections
        ) else []
        use_locator_gold = not use_chunk_gold and bool(resolved_locator_ids)
        evaluation_gold_ids = (
            row.relevant_chunk_ids if use_chunk_gold
            else resolved_locator_ids if use_locator_gold
            else _document_gold_ids(row)
        )
        gold_reference_mode = (
            "chunk" if use_chunk_gold else "locator" if use_locator_gold else "document"
        )
        retrieved_evaluation_ids: List[str] = []
        try:
            search_kwargs = {
                "retrieve_k": retrieval_settings.retrieve_k,
                "top_k_faiss": retrieval_settings.top_k_faiss,
                "hybrid_alpha": retrieval_settings.hybrid_alpha,
                "use_rerank": runtime_settings.features.enable_reranker,
            }
            if variant == "hybrid_current" and not record_trace:
                # Preserve the original P0-A benchmark path exactly when no
                # instrumentation or ablation was requested.
                retrieved, _ce_scores = idx.search(row.query, **search_kwargs)
            else:
                instrumented = idx.search_instrumented(
                    row.query,
                    variant=variant,
                    include_trace=(record_trace or variant in {
                        "anchor_scope", "anchor_scope_adaptive", "anchor_scope_adaptive_k"
                    }),
                    **search_kwargs,
                )
                retrieved, _ce_scores = instrumented.as_legacy()
                if instrumented.trace is not None:
                    retrieval_trace = instrumented.trace.to_dict()
            for rank, (score, meta) in enumerate(retrieved, start=1):
                retrieved_metas.append(meta)
                retrieved_evaluation_ids.append(
                    _chunk_uid(meta) if (use_chunk_gold or use_locator_gold) else _document_uid(meta)
                )
                retrieved_chunks.append(
                    RetrievedChunk(
                        rank=rank,
                        chunk_uid=_chunk_uid(meta),
                        score=float(score),
                        file=meta.get("file"),
                        path=meta.get("path"),
                        chunk_id=meta.get("chunk_id"),
                        source=meta.get("source"),
                        document_id=meta.get("document_id"),
                        page=meta.get("page"),
                        section=meta.get("section"),
                    )
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        latency_ms = (time.perf_counter() - started) * 1000.0
        retrieved_ids = retrieved_evaluation_ids
        retrieval_evaluated = row.answerable and error is None
        metrics = compute_query_metrics(
            retrieved_ids,
            evaluation_gold_ids,
            answerable=retrieval_evaluated,
            # Stable document/locator gold has no legacy graded chunk labels.
            # Use transparent binary relevance so NDCG remains measurable after
            # rechunking, without inventing different grades for the v2 chunks.
            relevance_grades=(
                row.relevance_grades
                if use_chunk_gold
                else {gold_id: 1.0 for gold_id in evaluation_gold_ids}
            ),
            ks=ks,
        )
        deduplicated_retrieved = list(dict.fromkeys(retrieved_ids))
        metrics["retrieved_candidates"] = float(len(retrieved_chunks))
        metrics["useful_candidates"] = float(
            len(set(deduplicated_retrieved) & set(evaluation_gold_ids))
        )
        metrics["latency_ms"] = float(latency_ms)
        context_blocks = clip_context_blocks(
            fuse_contiguous_passages([
                (retrieved_chunks[index].score, meta)
                for index, meta in enumerate(retrieved_metas)
            ]),
            max_chars=MAX_CONTEXT_CHARS,
            keep=FINAL_K,
        ) if retrieved_metas else []
        uid_to_meta = {_chunk_uid(meta): meta for meta in retrieved_metas}
        context_chunk_uids: List[str] = []
        for _, meta in context_blocks:
            fused_uids = meta.get("fused_chunk_uids") or [_chunk_uid(meta)]
            context_chunk_uids.extend(str(uid) for uid in fused_uids)
        context_chunk_uids = list(dict.fromkeys(context_chunk_uids))
        context_chars = sum(len(str(meta.get("text") or "")) for _, meta in context_blocks)
        context_evaluation_ids = [
            uid if (use_chunk_gold or use_locator_gold)
            else _document_uid(uid_to_meta.get(uid, {}))
            for uid in context_chunk_uids
        ]
        metrics["context_evidence_coverage"] = (
            float(len(set(context_evaluation_ids) & set(evaluation_gold_ids)))
            / len(set(evaluation_gold_ids))
            if evaluation_gold_ids else 0.0
        )
        metrics["context_chunks"] = float(len(context_chunk_uids))
        metrics["context_blocks"] = float(len(context_blocks))
        metrics["context_chars"] = float(context_chars)
        metrics["context_tokens_estimated"] = float(context_chars / 4.0)
        anchor_scope_data = (retrieval_trace or {}).get("anchor_scope") or {}
        sufficiency_data = (retrieval_trace or {}).get("sufficiency") or {}
        candidate_pool_ids = list(sufficiency_data.get("candidate_pool_ids") or [])
        candidate_pool_metas = [
            idx.metas[candidate_id]
            for candidate_id in candidate_pool_ids
            if isinstance(candidate_id, int) and 0 <= candidate_id < len(idx.metas)
        ]
        candidate_pool_chunk_uids = [_chunk_uid(meta) for meta in candidate_pool_metas]
        pool_evaluation_ids = [
            _chunk_uid(meta) if (use_chunk_gold or use_locator_gold) else _document_uid(meta)
            for meta in candidate_pool_metas
        ]
        premature_stop, premature_stop_eligible = is_premature_stop(
            selected_ids=context_evaluation_ids,
            available_ids=pool_evaluation_ids,
            gold_ids=evaluation_gold_ids,
        )
        premature_stop_eligible = bool(row.answerable and premature_stop_eligible)
        premature_stop = bool(premature_stop_eligible and premature_stop)
        metrics["premature_stop"] = float(premature_stop)
        metrics["premature_stop_eligible"] = float(premature_stop_eligible)
        metrics["sufficiency_ms"] = float(sufficiency_data.get("timing_ms") or 0.0)
        anchor_count = len(anchor_scope_data.get("selected_anchor_ids") or [])
        scope_chunk_count = len(anchor_scope_data.get("selected_scope_ids") or [])
        metrics["multiple_anchors"] = float(anchor_count > 1)
        metrics["multi_chunk_scope"] = float(scope_chunk_count > 0)
        metrics["scope_triggered"] = float(bool(anchor_scope_data.get("scope_triggered")))
        metrics["rescored_candidates"] = float(
            anchor_scope_data.get("rescored_candidates") or 0
        )
        scope_timings = anchor_scope_data.get("timings_ms") or {}
        for timing_name in (
            "anchor_selection", "structural_collection", "expansion_scoring",
            "scope_reranking", "total",
        ):
            metrics[f"scope_{timing_name}_ms"] = float(scope_timings.get(timing_name) or 0.0)
        results.append(
            QueryEvaluation(
                query_id=row.query_id,
                query=row.query,
                expected_answer=row.expected_answer,
                question_type=row.question_type,
                difficulty=row.difficulty,
                answerable=row.answerable,
                relevant_document=row.relevant_document,
                relevant_chunk_ids=evaluation_gold_ids,
                gold_reference_mode=gold_reference_mode,
                required_facts=row.required_facts,
                retrieved=retrieved_chunks,
                metrics=metrics,
                retrieval_evaluated=retrieval_evaluated,
                latency_ms=latency_ms,
                retrieval_trace=retrieval_trace,
                context_chunk_uids=context_chunk_uids,
                context_block_count=len(context_blocks),
                context_chars=context_chars,
                candidate_pool_chunk_uids=candidate_pool_chunk_uids,
                sufficiency_decision=sufficiency_data.get("final_decision"),
                anchor_count=anchor_count,
                scope_chunk_count=scope_chunk_count,
                error=error,
            )
        )

    index_files = {
        "corpus_sha256": _sha256(Path(idx.corpus_path)),
        "embeddings_sha256": _sha256(Path(idx.emb_path)),
        "faiss_sha256": _sha256(Path(idx.faiss_path)),
    }
    effective_configuration = {
        **effective_runtime_configuration(runtime_settings, device=DEVICE),
        "pipeline": "current",
        "retrieval_variant": variant,
        "record_trace": record_trace,
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "index_chunks": len(getattr(idx, "metas", []) or []),
        **index_files,
    }
    metrics_global = average_metrics(
        item.metrics for item in results if item.retrieval_evaluated
    )
    eligible_stops = sum(
        item.metrics.get("premature_stop_eligible", 0.0) for item in results
    )
    metrics_global["premature_stop_rate"] = (
        sum(item.metrics.get("premature_stop", 0.0) for item in results) / eligible_stops
        if eligible_stops else 0.0
    )
    report = BenchmarkReport(
        benchmark_name=str(config.get("benchmark_name", "auxilium-current-baseline")),
        commit_git=_git_commit(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        configuration=effective_configuration,
        embedding_model=retrieval_settings.embedding_model,
        reranker=getattr(idx, "rerank_model_used", None),
        query_count=len(results),
        retrieval_evaluated_queries=sum(item.retrieval_evaluated for item in results),
        unanswerable_queries=sum(not item.answerable for item in results),
        failed_queries=sum(item.error is not None for item in results),
        metrics_global=metrics_global,
        metrics_by_question_type=aggregate_by_question_type(results),
        results=results,
    )

    if output_override:
        output_path = output_override
    else:
        output_dir = _resolve_from_back(str(config.get("output_dir", "eval/reports")))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = output_dir / f"{variant}-{stamp}.json"
    return write_report(report, output_path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Auxilium's unchanged current retrieval pipeline."
    )
    parser.add_argument(
        "--config",
        default="eval/configs/current.yaml",
        help="Path to the JSON-compatible YAML benchmark config.",
    )
    parser.add_argument("--output", help="Optional exact report output path.")
    parser.add_argument(
        "--variant",
        choices=(
            "dense_only",
            "bm25_only",
            "hybrid_current",
            "hybrid_no_mmr",
            "hybrid_no_reranker",
            "anchor_scope",
            "anchor_scope_adaptive",
            "anchor_scope_adaptive_k",
        ),
        help="Benchmark-only retrieval variant (defaults to the config value).",
    )
    parser.add_argument(
        "--record-trace",
        action="store_true",
        default=None,
        help="Record all retrieval stages for each query.",
    )
    args = parser.parse_args()

    config_path = _resolve_from_back(args.config).resolve()
    output_path = Path(args.output).resolve() if args.output else None
    written = run(
        config_path,
        output_path,
        variant_override=args.variant,
        record_trace_override=args.record_trace,
    )
    print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
