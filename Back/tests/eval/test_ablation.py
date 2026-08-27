import unittest
from pathlib import Path

from pydantic import ValidationError

from eval.ablation import (
    abstention_metrics,
    build_comparative_report,
    latency_summary,
    metric_deltas,
    percentile,
)
from eval.dataset import DatasetError, load_jsonl, validate_relevant_chunk_ids
from eval.schemas import BenchmarkQuery, BenchmarkReport, QueryEvaluation, RetrievedChunk


def _query_result(
    query_id,
    retrieved_ids,
    metrics,
    *,
    answerable=True,
    latency_ms=10.0,
):
    return QueryEvaluation(
        query_id=query_id,
        query=f"Question {query_id}",
        expected_answer="answer" if answerable else None,
        question_type="factuelle_simple" if answerable else "non_repondable",
        difficulty="easy",
        answerable=answerable,
        relevant_chunk_ids=["gold::0"] if answerable else [],
        required_facts=[],
        retrieved=[
            RetrievedChunk(rank=rank, chunk_uid=chunk_id, score=1.0 / rank)
            for rank, chunk_id in enumerate(retrieved_ids, start=1)
        ],
        metrics=metrics,
        retrieval_evaluated=answerable,
        latency_ms=latency_ms,
    )


def _report(results):
    return BenchmarkReport(
        benchmark_name="test",
        commit_git="abc",
        timestamp_utc="2026-01-01T00:00:00Z",
        configuration={},
        embedding_model="embed",
        reranker="reranker",
        query_count=len(results),
        retrieval_evaluated_queries=sum(result.answerable for result in results),
        unanswerable_queries=sum(not result.answerable for result in results),
        failed_queries=0,
        metrics_global={"recall@1": 1.0, "mrr": 1.0},
        metrics_by_question_type={},
        results=results,
    )


class AblationAggregationTests(unittest.TestCase):
    def test_percentiles_and_latency_summary(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 95), 3.85)
        results = [
            _query_result("q1", ["gold::0"], {"recall@1": 1.0}, latency_ms=1),
            _query_result("q2", ["gold::0"], {"recall@1": 1.0}, latency_ms=3),
        ]
        summary = latency_summary(results)
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["mean_ms"], 2.0)
        self.assertEqual(summary["p50_ms"], 2.0)

    def test_metric_deltas_preserve_unavailable_metrics(self):
        deltas = metric_deltas(
            {"recall@1": 0.75, "ndcg@1": None},
            {"recall@1": 0.5, "ndcg@1": 1.0},
        )
        self.assertEqual(deltas["recall@1"], 0.25)
        self.assertIsNone(deltas["ndcg@1"])

    def test_abstention_metrics_are_prepared_but_not_fabricated(self):
        metrics = abstention_metrics(
            [_query_result("q1", ["noise::0"], {}, answerable=False)]
        )
        self.assertEqual(metrics["eligible_queries"], 1)
        self.assertEqual(metrics["evaluated_queries"], 0)
        self.assertIsNone(metrics["abstention_accuracy"])
        self.assertEqual(metrics["status"], "prepared_not_evaluated")

    def test_report_contains_global_and_per_query_differences(self):
        baseline_result = _query_result(
            "q1", ["gold::0", "noise::0"], {"recall@1": 1.0, "mrr": 1.0}
        )
        dense_result = _query_result(
            "q1", ["noise::0", "gold::0"], {"recall@1": 0.0, "mrr": 0.5}
        )
        baseline = _report([baseline_result])
        dense = _report([dense_result])
        dense.metrics_global = {"recall@1": 0.0, "mrr": 0.5}
        report = build_comparative_report(
            {"hybrid_current": baseline, "dense_only": dense},
            benchmark_name="comparison",
            timestamp_utc="2026-01-01T00:00:00Z",
            configuration={},
        )
        self.assertEqual(
            report["variants"]["dense_only"]["metric_deltas_vs_hybrid_current"][
                "recall@1"
            ],
            -1.0,
        )
        dense_query = report["query_comparisons"][0]["variants"]["dense_only"]
        self.assertEqual(dense_query["first_relevant_rank"], 2)
        self.assertEqual(dense_query["first_relevant_rank_delta_vs_hybrid_current"], 1)
        self.assertFalse(dense_query["same_ranking_as_hybrid_current"])


class DatasetQualityTests(unittest.TestCase):
    def test_answerable_rows_require_gold_chunks(self):
        with self.assertRaises(ValidationError):
            BenchmarkQuery(
                query_id="invalid",
                query="Question",
                expected_answer="Answer",
                relevant_chunk_ids=[],
                answerable=True,
                question_type="factuelle_simple",
            )

    def test_gold_chunk_validation_fails_on_unknown_ids(self):
        row = BenchmarkQuery(
            query_id="q1",
            query="Question",
            expected_answer="Answer",
            relevant_chunk_ids=["known::0"],
            answerable=True,
            question_type="factuelle_simple",
        )
        with self.assertRaises(DatasetError):
            validate_relevant_chunk_ids([row], {"other::0"})

    def test_pilot_is_valid_and_covers_required_categories(self):
        pilot_path = Path(__file__).resolve().parents[2] / "eval" / "data" / "pilot.jsonl"
        rows = load_jsonl(pilot_path)
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(row.answerable for row in rows), 17)
        self.assertEqual(sum(not row.answerable for row in rows), 3)
        types = {row.question_type for row in rows}
        self.assertTrue(
            {
                "factuelle_simple",
                "paraphrase",
                "multi_passage",
                "distracteurs_semantiques",
                "exhaustive",
                "exhaustive_multi_documents",
                "non_repondable",
                "email_metadata",
                "comparative_multi_documents",
            }.issubset(types)
        )


if __name__ == "__main__":
    unittest.main()
