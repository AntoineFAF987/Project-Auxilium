from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .ablation import build_comparative_report
from .dataset import load_jsonl
from .run_current import _load_config, _resolve_from_back, run as run_variant
from .schemas import BenchmarkReport


DEFAULT_VARIANTS = [
    "dense_only",
    "bm25_only",
    "hybrid_current",
    "hybrid_no_mmr",
    "hybrid_no_reranker",
]


def _load_ablation_config(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON-compatible YAML config {path}: {exc}") from exc
    variants = payload.get("variants", DEFAULT_VARIANTS)
    from rag_core.contracts import RETRIEVAL_VARIANTS
    if not variants or any(variant not in RETRIEVAL_VARIANTS for variant in variants):
        raise ValueError("Ablation variants must be non-empty known retrieval variants")
    if len(variants) != len(set(variants)):
        raise ValueError("Ablation variants must not contain duplicates")
    if "hybrid_current" not in variants:
        raise ValueError("Ablation variants must include hybrid_current as baseline")
    return payload


def _warm_up(variant: str, query: str) -> None:
    from api.index_singleton import idx
    from runtime_settings import get_runtime_settings

    settings = get_runtime_settings()
    kwargs = {
        "retrieve_k": settings.retrieval.retrieve_k,
        "top_k_faiss": settings.retrieval.top_k_faiss,
        "hybrid_alpha": settings.retrieval.hybrid_alpha,
        "use_rerank": settings.features.enable_reranker,
    }
    if variant == "hybrid_current":
        idx.search(query, **kwargs)
    else:
        idx.search_instrumented(
            query, variant=variant, include_trace=False, **kwargs
        )


def run(config_path: Path, output_override: Path | None = None) -> Path:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    config = _load_ablation_config(config_path)
    base_config_path = _resolve_from_back(str(config["base_config"])).resolve()
    base_config = _load_config(base_config_path)
    dataset_path = _resolve_from_back(str(base_config["dataset"])).resolve()
    rows = load_jsonl(dataset_path)
    variants: List[str] = list(config["variants"])
    warmup = bool(config.get("warmup", True))

    reports: Dict[str, BenchmarkReport] = {}
    with tempfile.TemporaryDirectory(prefix="auxilium-ablation-") as temp_dir:
        temporary_root = Path(temp_dir)
        for variant in variants:
            if warmup:
                _warm_up(variant, rows[0].query)
            report_path = temporary_root / f"{variant}.json"
            run_variant(
                base_config_path,
                report_path,
                variant_override=variant,
                record_trace_override=False,
            )
            reports[variant] = BenchmarkReport.model_validate_json(
                report_path.read_text(encoding="utf-8")
            )

    timestamp = datetime.now(timezone.utc)
    comparison = build_comparative_report(
        reports,
        benchmark_name=str(config.get("benchmark_name", "auxilium-retrieval-ablations")),
        timestamp_utc=timestamp.isoformat(),
        configuration={
            "base_config": str(base_config_path),
            "dataset": str(dataset_path),
            "dataset_sha256": reports["hybrid_current"].configuration.get(
                "dataset_sha256"
            ),
            "variants": variants,
            "warmup_per_variant": warmup,
            "variant_traces": False,
            "annotation_policy": config.get("annotation_policy"),
            "effective_retrieval": reports["hybrid_current"].configuration,
        },
    )

    if output_override is not None:
        output_path = output_override
    else:
        output_dir = _resolve_from_back(str(config.get("output_dir", "eval/reports")))
        stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
        output_path = output_dir / f"retrieval-ablations-{stamp}.json"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Auxilium retrieval variants on one active index."
    )
    parser.add_argument(
        "--config",
        default="eval/configs/ablations.yaml",
        help="Path to the JSON-compatible YAML ablation config.",
    )
    parser.add_argument("--output", help="Optional exact comparative report path.")
    args = parser.parse_args()

    config_path = _resolve_from_back(args.config).resolve()
    output_path = Path(args.output).resolve() if args.output else None
    written = run(config_path, output_path)
    print(written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
