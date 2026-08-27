import math
import unittest

from eval.metrics_retrieval import (
    average_metrics,
    compute_query_metrics,
    ndcg_at_k,
)


class RetrievalMetricsTests(unittest.TestCase):
    def test_first_relevant_result_at_rank_one(self):
        metrics = compute_query_metrics(
            ["a", "b"], ["a"], answerable=True, ks=(1, 3)
        )
        self.assertEqual(metrics["recall@1"], 1.0)
        self.assertEqual(metrics["precision@1"], 1.0)
        self.assertEqual(metrics["hit_rate@1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_first_relevant_result_at_rank_two(self):
        metrics = compute_query_metrics(
            ["noise", "gold", "other"], ["gold"], answerable=True, ks=(1, 3)
        )
        self.assertEqual(metrics["recall@1"], 0.0)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertEqual(metrics["hit_rate@1"], 0.0)
        self.assertEqual(metrics["mrr"], 0.5)

    def test_multiple_relevant_passages(self):
        metrics = compute_query_metrics(
            ["a", "noise", "c"], ["a", "c"], answerable=True, ks=(1, 3)
        )
        self.assertEqual(metrics["recall@1"], 0.5)
        self.assertEqual(metrics["recall@3"], 1.0)
        self.assertTrue(math.isclose(metrics["precision@3"], 2 / 3))
        self.assertEqual(metrics["mrr"], 1.0)

    def test_no_relevant_result_retrieved(self):
        metrics = compute_query_metrics(
            ["a", "b"], ["missing"], answerable=True, ks=(1, 3)
        )
        self.assertEqual(metrics["recall@3"], 0.0)
        self.assertEqual(metrics["precision@3"], 0.0)
        self.assertEqual(metrics["hit_rate@3"], 0.0)
        self.assertEqual(metrics["mrr"], 0.0)

    def test_unanswerable_query_is_not_averaged_as_retrieval_failure(self):
        unanswerable = compute_query_metrics(
            ["a", "b"], [], answerable=False, ks=(1, 3)
        )
        answerable = compute_query_metrics(
            ["gold"], ["gold"], answerable=True, ks=(1, 3)
        )
        self.assertTrue(all(value is None for value in unanswerable.values()))
        averages = average_metrics([unanswerable, answerable])
        self.assertEqual(averages["recall@1"], 1.0)
        self.assertEqual(averages["mrr"], 1.0)

    def test_ndcg_uses_explicit_relevance_grades(self):
        ideal = ndcg_at_k(["high", "medium"], {"high": 3, "medium": 1}, 2)
        reversed_order = ndcg_at_k(
            ["medium", "high"], {"high": 3, "medium": 1}, 2
        )
        self.assertEqual(ideal, 1.0)
        self.assertIsNotNone(reversed_order)
        self.assertLess(reversed_order, ideal)
        self.assertIsNone(ndcg_at_k(["a"], {}, 1))


if __name__ == "__main__":
    unittest.main()
