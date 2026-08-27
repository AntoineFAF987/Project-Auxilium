from __future__ import annotations

import math
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .schemas import BenchmarkReport, QueryEvaluation


def percentile(values: Sequence[float], percentile_value: float) -> Optional[float]:
    """Linear percentile without adding a numerical dependency to eval code."""

    if not values:
        return None
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def latency_summary(results: Iterable[QueryEvaluation]) -> Dict[str, Optional[float]]:
    values = [float(result.latency_ms) for result in results if result.error is None]
    return {
        "sample_count": len(values),
        "mean_ms": mean(values) if values else None,
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
    }


def metric_deltas(
    metrics: Mapping[str, Optional[float]],
    baseline: Mapping[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    names = sorted(set(metrics) | set(baseline))
    return {
        name: (
            float(metrics[name]) - float(baseline[name])
            if metrics.get(name) is not None and baseline.get(name) is not None
            else None
        )
        for name in names
    }


def first_relevant_rank(result: QueryEvaluation) -> Optional[int]:
    relevant = set(result.relevant_chunk_ids)
    for retrieved in result.retrieved:
        if retrieved.chunk_uid in relevant:
            return retrieved.rank
    return None


def abstention_metrics(results: Iterable[QueryEvaluation]) -> Dict[str, Any]:
    non_answerable = [result for result in results if not result.answerable]
    return {
        "eligible_queries": len(non_answerable),
        "evaluated_queries": 0,
        "correct_abstentions": None,
        "false_answers": None,
        "abstention_accuracy": None,
        "false_answer_rate": None,
        "retrieval_returned_candidates": sum(bool(result.retrieved) for result in non_answerable),
        "status": "prepared_not_evaluated",
        "reason": "The retrieval runner does not execute the runtime abstention decision.",
    }


def build_comparative_report(
    reports: Mapping[str, BenchmarkReport],
    *,
    benchmark_name: str,
    timestamp_utc: str,
    configuration: Mapping[str, Any],
    baseline_variant: str = "hybrid_current",
) -> Dict[str, Any]:
    if baseline_variant not in reports:
        raise ValueError(f"Missing baseline variant: {baseline_variant}")

    baseline = reports[baseline_variant]
    baseline_by_query = {result.query_id: result for result in baseline.results}
    variant_summaries: Dict[str, Any] = {}
    for variant, report in reports.items():
        variant_summaries[variant] = {
            "query_count": report.query_count,
            "retrieval_evaluated_queries": report.retrieval_evaluated_queries,
            "failed_queries": report.failed_queries,
            "metrics_global": report.metrics_global,
            "metric_deltas_vs_hybrid_current": metric_deltas(
                report.metrics_global, baseline.metrics_global
            ),
            "metrics_by_question_type": report.metrics_by_question_type,
            "latency": latency_summary(report.results),
            "abstention": abstention_metrics(report.results),
        }

    query_comparisons: List[Dict[str, Any]] = []
    for query_id, baseline_result in baseline_by_query.items():
        variants: Dict[str, Any] = {}
        baseline_ids = [item.chunk_uid for item in baseline_result.retrieved]
        baseline_first_rank = first_relevant_rank(baseline_result)
        for variant, report in reports.items():
            result_by_query = {item.query_id: item for item in report.results}
            if query_id not in result_by_query:
                raise ValueError(f"Variant {variant} is missing query {query_id}")
            result = result_by_query[query_id]
            retrieved_ids = [item.chunk_uid for item in result.retrieved]
            first_rank = first_relevant_rank(result)
            variants[variant] = {
                "retrieved_chunk_ids": retrieved_ids,
                "first_relevant_rank": first_rank,
                "first_relevant_rank_delta_vs_hybrid_current": (
                    first_rank - baseline_first_rank
                    if first_rank is not None and baseline_first_rank is not None
                    else None
                ),
                "same_ranking_as_hybrid_current": retrieved_ids == baseline_ids,
                "metrics": result.metrics,
                "metric_deltas_vs_hybrid_current": metric_deltas(
                    result.metrics, baseline_result.metrics
                ),
                "latency_ms": result.latency_ms,
                "error": result.error,
            }
        query_comparisons.append(
            {
                "query_id": query_id,
                "query": baseline_result.query,
                "question_type": baseline_result.question_type,
                "difficulty": baseline_result.difficulty,
                "answerable": baseline_result.answerable,
                "relevant_chunk_ids": baseline_result.relevant_chunk_ids,
                "variants": variants,
            }
        )

    question_types = sorted({result.question_type for result in baseline.results})
    return {
        "schema_version": 1,
        "benchmark_name": benchmark_name,
        "commit_git": baseline.commit_git,
        "timestamp_utc": timestamp_utc,
        "baseline_variant": baseline_variant,
        "configuration": dict(configuration),
        "dataset": {
            "query_count": baseline.query_count,
            "answerable_queries": baseline.query_count - baseline.unanswerable_queries,
            "unanswerable_queries": baseline.unanswerable_queries,
            "question_types": question_types,
        },
        "embedding_model": baseline.embedding_model,
        "reranker": baseline.reranker,
        "variants": variant_summaries,
        "query_comparisons": query_comparisons,
    }
