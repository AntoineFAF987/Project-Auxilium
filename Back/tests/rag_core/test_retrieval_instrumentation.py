import unittest

import numpy as np

from rag_core.contracts import RETRIEVAL_VARIANTS
from rag_core.indexer import RAGIndexer


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeEmbeddingModel:
    def __init__(self, query_vectors):
        self.query_vectors = query_vectors

    def encode(self, query, **_kwargs):
        return _FakeTensor(self.query_vectors[query])


class _FakeFaissIndex:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def search(self, query, k):
        scores = self.embeddings @ query[0]
        ids = np.argsort(scores)[::-1][:k]
        return (
            np.asarray([scores[ids]], dtype=np.float32),
            np.asarray([ids], dtype=np.int64),
        )


class _FakeBm25:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = [set(tokens) for tokens in tokenized_texts]

    def get_scores(self, query_tokens):
        query = set(query_tokens)
        return np.asarray(
            [len(query & tokens) for tokens in self.tokenized_texts],
            dtype=np.float32,
        )


class _FakeCrossEncoder:
    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            query_tokens = set(query.lower().split())
            text_tokens = set(text.lower().split())
            scores.append(float(len(query_tokens & text_tokens)) + len(text) / 10_000)
        return np.asarray(scores, dtype=np.float32)


class _FailingCrossEncoder:
    def predict(self, pairs):
        raise RuntimeError("deliberate reranker failure")


def _normalise(rows):
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _make_indexer():
    texts = [
        "alpha contract renewal date june",
        "beta pricing and contract clauses",
        "alpha project delivery schedule",
        "unrelated cooking recipe",
        "alpha contract exact wording",
        "project risks and mitigations",
    ]
    embeddings = _normalise(
        [
            [1.0, 0.1, 0.0],
            [0.8, 0.5, 0.0],
            [0.7, 0.0, 0.7],
            [0.0, 0.1, 1.0],
            [0.9, 0.2, 0.1],
            [0.2, 0.0, 0.9],
        ]
    )
    query_vectors = {
        "alpha contract": _normalise([[1.0, 0.2, 0.0]])[0],
        "project risks": _normalise([[0.2, 0.0, 1.0]])[0],
        "pricing clauses": _normalise([[0.6, 0.8, 0.0]])[0],
    }

    indexer = object.__new__(RAGIndexer)
    indexer.metas = [
        {
            "file": f"doc-{idx}.txt",
            "path": f"C:/corpus/doc-{idx}.txt",
            "chunk_id": idx,
            "source": "local",
            "fingerprint": f"fp-{idx}",
        }
        for idx in range(len(texts))
    ]
    indexer.corpus = [
        {**meta, "text": text} for meta, text in zip(indexer.metas, texts)
    ]
    indexer.texts = texts
    indexer.embeddings = embeddings
    indexer.embed_model = _FakeEmbeddingModel(query_vectors)
    indexer.faiss_index = _FakeFaissIndex(embeddings)
    indexer.bm25_index = _FakeBm25([text.split() for text in texts])
    indexer.cross_encoder = _FakeCrossEncoder()
    indexer.rerank_model_used = "fake-cross-encoder"
    return indexer


class RetrievalInstrumentationParityTests(unittest.TestCase):
    def assert_legacy_equal(self, expected, actual):
        expected_items, expected_ce = expected
        actual_items, actual_ce = actual
        self.assertEqual(expected_ce, actual_ce)
        self.assertEqual(len(expected_items), len(actual_items))
        for expected_item, actual_item in zip(expected_items, actual_items):
            self.assertEqual(expected_item[0], actual_item[0])
            self.assertEqual(expected_item[1], actual_item[1])

    def test_current_variant_matches_search_for_multiple_queries(self):
        indexer = _make_indexer()
        for query in ("alpha contract", "project risks", "pricing clauses"):
            with self.subTest(query=query):
                kwargs = {
                    "retrieve_k": 3,
                    "top_k_faiss": 5,
                    "hybrid_alpha": 0.8,
                    "use_rerank": True,
                }
                legacy = indexer.search(query, **kwargs)
                instrumented = indexer.search_instrumented(
                    query,
                    variant="hybrid_current",
                    include_trace=True,
                    **kwargs,
                )
                self.assert_legacy_equal(legacy, instrumented.as_legacy())

    def test_enabling_trace_does_not_change_result(self):
        indexer = _make_indexer()
        kwargs = {
            "retrieve_k": 3,
            "top_k_faiss": 5,
            "hybrid_alpha": 0.8,
            "use_rerank": True,
            "variant": "hybrid_current",
        }
        without_trace = indexer.search_instrumented(
            "alpha contract", include_trace=False, **kwargs
        )
        with_trace = indexer.search_instrumented(
            "alpha contract", include_trace=True, **kwargs
        )
        self.assertIsNone(without_trace.trace)
        self.assertIsNotNone(with_trace.trace)
        self.assert_legacy_equal(without_trace.as_legacy(), with_trace.as_legacy())

    def test_current_variant_matches_search_without_reranker(self):
        indexer = _make_indexer()
        indexer.cross_encoder = None
        kwargs = {
            "retrieve_k": 3,
            "top_k_faiss": 5,
            "hybrid_alpha": 0.8,
            "use_rerank": True,
        }
        legacy = indexer.search("project risks", **kwargs)
        instrumented = indexer.search_instrumented(
            "project risks",
            variant="hybrid_current",
            include_trace=True,
            **kwargs,
        )
        self.assert_legacy_equal(legacy, instrumented.as_legacy())

    def test_current_variant_matches_search_on_reranker_failure(self):
        indexer = _make_indexer()
        indexer.cross_encoder = _FailingCrossEncoder()
        kwargs = {
            "retrieve_k": 3,
            "top_k_faiss": 5,
            "hybrid_alpha": 0.8,
            "use_rerank": True,
        }
        legacy = indexer.search("project risks", **kwargs)
        instrumented = indexer.search_instrumented(
            "project risks",
            variant="hybrid_current",
            include_trace=True,
            **kwargs,
        )
        self.assert_legacy_equal(legacy, instrumented.as_legacy())
        self.assertEqual(len(instrumented.trace.errors), 1)

    def test_trace_contains_scores_ranks_metadata_and_all_stages(self):
        result = _make_indexer().search_instrumented(
            "alpha contract",
            retrieve_k=3,
            top_k_faiss=5,
            hybrid_alpha=0.8,
            use_rerank=True,
            variant="hybrid_current",
            include_trace=True,
        )
        trace = result.trace
        self.assertIsNotNone(trace)
        self.assertTrue(trace.dense_candidate_ids)
        self.assertTrue(trace.bm25_candidate_ids)
        self.assertTrue(trace.union_candidate_ids)
        self.assertTrue(trace.fusion_candidate_ids)
        self.assertTrue(trace.top_pool_candidate_ids)
        self.assertTrue(trace.mmr_selected_ids)
        self.assertTrue(trace.reranked_candidate_ids)
        self.assertEqual(trace.reranked_candidate_ids, trace.final_candidate_ids)

        by_id = {candidate.candidate_id: candidate for candidate in trace.candidates}
        final = by_id[trace.final_candidate_ids[0]]
        self.assertIn("file", final.metadata)
        self.assertIsNotNone(final.hybrid_score)
        self.assertIsNotNone(final.mmr_rank)
        self.assertIsNotNone(final.reranker_score)
        self.assertEqual(final.final_rank, 1)

    def test_all_benchmark_variants_are_executable(self):
        for variant in sorted(RETRIEVAL_VARIANTS):
            with self.subTest(variant=variant):
                result = _make_indexer().search_instrumented(
                    "alpha contract",
                    retrieve_k=3,
                    top_k_faiss=5,
                    hybrid_alpha=0.8,
                    use_rerank=True,
                    variant=variant,
                    include_trace=True,
                )
                self.assertLessEqual(len(result.items), 3)
                self.assertEqual(result.trace.variant, variant)


if __name__ == "__main__":
    unittest.main()
