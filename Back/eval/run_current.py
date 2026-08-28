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

    runtime_settings = get_runtime_settings()
    retrieval_settings = runtime_settings.retrieval

    if variant not in RETRIEVAL_VARIANTS:
        choices = ", ".join(sorted(RETRIEVAL_VARIANTS))
        raise ValueError(f"Unknown retrieval variant {variant!r}; choose one of: {choices}")

    validate_relevant_chunk_ids(rows, (_chunk_uid(meta) for meta in idx.metas))

    results: List[QueryEvaluation] = []
    for row in rows:
        started = time.perf_counter()
        retrieved_chunks: List[RetrievedChunk] = []
        retrieval_trace = None
        error = None
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
                    include_trace=record_trace,
                    **search_kwargs,
                )
                retrieved, _ce_scores = instrumented.as_legacy()
                if instrumented.trace is not None:
                    retrieval_trace = instrumented.trace.to_dict()
            for rank, (score, meta) in enumerate(retrieved, start=1):
                retrieved_chunks.append(
                    RetrievedChunk(
                        rank=rank,
                        chunk_uid=_chunk_uid(meta),
                        score=float(score),
                        file=meta.get("file"),
                        path=meta.get("path"),
                        chunk_id=meta.get("chunk_id"),
                        source=meta.get("source"),
                    )
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        latency_ms = (time.perf_counter() - started) * 1000.0
        retrieved_ids = [item.chunk_uid for item in retrieved_chunks]
        retrieval_evaluated = row.answerable and error is None
        metrics = compute_query_metrics(
            retrieved_ids,
            row.relevant_chunk_ids,
            answerable=retrieval_evaluated,
            relevance_grades=row.relevance_grades,
            ks=ks,
        )
        results.append(
            QueryEvaluation(
                query_id=row.query_id,
                query=row.query,
                expected_answer=row.expected_answer,
                question_type=row.question_type,
                difficulty=row.difficulty,
                answerable=row.answerable,
                relevant_document=row.relevant_document,
                relevant_chunk_ids=row.relevant_chunk_ids,
                required_facts=row.required_facts,
                retrieved=retrieved_chunks,
                metrics=metrics,
                retrieval_evaluated=retrieval_evaluated,
                latency_ms=latency_ms,
                retrieval_trace=retrieval_trace,
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
        metrics_global=average_metrics(
            item.metrics for item in results if item.retrieval_evaluated
        ),
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
