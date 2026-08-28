import unittest
import os
from pathlib import Path
from unittest.mock import Mock, patch


class RuntimeSettingsLoadingTests(unittest.TestCase):
    def test_defaults_match_the_current_runtime_baseline(self):
        from runtime_settings import load_runtime_settings

        settings = load_runtime_settings(config_data={}, env={})

        self.assertEqual(settings.retrieval.embedding_model, "paraphrase-multilingual-mpnet-base-v2")
        self.assertEqual(settings.retrieval.reranker_model, "BAAI/bge-reranker-v2-m3")
        self.assertEqual(
            settings.retrieval.reranker_fallback,
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        self.assertEqual(settings.retrieval.retrieve_k, 12)
        self.assertEqual(settings.retrieval.top_k_faiss, 100)
        self.assertEqual(settings.retrieval.top_k_bm25, 100)
        self.assertEqual(settings.retrieval.final_k, 10)
        self.assertEqual(settings.retrieval.max_context_chars, 12000)
        self.assertEqual(settings.retrieval.hybrid_alpha, 0.8)
        self.assertEqual(settings.retrieval.exact_match_bonus, 0.15)
        self.assertEqual(settings.retrieval.mmr_lambda, 0.7)
        self.assertEqual(settings.generation.model, "mistral-small-latest")
        self.assertEqual(settings.generation.temperature, 0.6)
        self.assertEqual(settings.generation.strict_temperature, 0.45)
        self.assertEqual(settings.generation.max_tokens, 1200)

    def test_config_file_values_override_defaults(self):
        from runtime_settings import load_runtime_settings

        settings = load_runtime_settings(
            config_data={
                "retrieval": {
                    "retrieve_k": 7,
                    "hybrid_alpha": 0.55,
                    "exact_match_bonus": 0.2,
                },
                "rag_params": {"faithfulness_threshold": 0.66},
                "llm": {
                    "provider": "mistral",
                    "model": "custom-model",
                    "temperature": 0.3,
                    "max_tokens": 777,
                },
            },
            env={},
        )

        self.assertEqual(settings.retrieval.retrieve_k, 7)
        self.assertEqual(settings.retrieval.hybrid_alpha, 0.55)
        self.assertEqual(settings.retrieval.exact_match_bonus, 0.2)
        self.assertEqual(settings.thresholds.faithfulness, 0.66)
        self.assertEqual(settings.generation.model, "custom-model")
        self.assertEqual(settings.generation.temperature, 0.3)
        self.assertEqual(settings.generation.max_tokens, 777)

    def test_environment_values_override_the_config_file(self):
        from runtime_settings import load_runtime_settings

        settings = load_runtime_settings(
            config_data={
                "retrieval": {"retrieve_k": 7, "hybrid_alpha": 0.55},
                "llm": {"model": "file-model"},
            },
            env={
                "AUXILIUM_RETRIEVE_K": "9",
                "AUXILIUM_HYBRID_ALPHA": "0.4",
                "AUXILIUM_LLM_MODEL": "env-model",
                "AUXILIUM_ENABLE_QUERY_EXPANSION": "false",
            },
        )

        self.assertEqual(settings.retrieval.retrieve_k, 9)
        self.assertEqual(settings.retrieval.hybrid_alpha, 0.4)
        self.assertEqual(settings.generation.model, "env-model")
        self.assertFalse(settings.features.enable_query_expansion)

    def test_explicit_experiment_override_has_highest_priority(self):
        from runtime_settings import load_runtime_settings

        settings = load_runtime_settings(
            config_data={"retrieval": {"retrieve_k": 7}},
            env={"AUXILIUM_RETRIEVE_K": "9"},
            overrides={"retrieval": {"retrieve_k": 4}},
        )

        self.assertEqual(settings.retrieval.retrieve_k, 4)

    def test_invalid_values_are_rejected(self):
        from pydantic import ValidationError
        from runtime_settings import load_runtime_settings

        invalid_configs = [
            {"retrieval": {"retrieve_k": 0}},
            {"retrieval": {"hybrid_alpha": 1.1}},
            {"retrieval": {"mmr_lambda": -0.1}},
            {"llm": {"provider": "decorative-provider"}},
            {"llm": {"max_tokens": 0}},
            {"llm": {"stream": True}},
            {"rag_params": {"context_relevance_threshold": 2}},
            {"roots": []},
            {"directories": []},
        ]
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises((ValidationError, ValueError)):
                load_runtime_settings(config_data=config, env={})

    def test_configuration_hash_is_stable_and_value_sensitive(self):
        from runtime_settings import load_runtime_settings

        first = load_runtime_settings(config_data={}, env={})
        same = load_runtime_settings(config_data={}, env={})
        changed = load_runtime_settings(
            config_data={"retrieval": {"retrieve_k": 11}},
            env={},
        )

        self.assertEqual(first.config_hash(), same.config_hash())
        self.assertNotEqual(first.config_hash(), changed.config_hash())

    def test_only_generation_settings_are_hot_reloadable(self):
        from runtime_settings import load_runtime_settings

        current = load_runtime_settings(config_data={}, env={})
        generation_change = load_runtime_settings(
            config_data={"llm": {"temperature": 0.25}},
            env={},
        )
        retrieval_change = load_runtime_settings(
            config_data={"retrieval": {"retrieve_k": 8}},
            env={},
        )

        self.assertFalse(generation_change.requires_restart_compared_to(current))
        self.assertTrue(retrieval_change.requires_restart_compared_to(current))


class RuntimeSettingsConsumerTests(unittest.TestCase):
    def test_generation_payload_uses_effective_front_editable_values(self):
        from rag_core.llm import ask_mistral_with_context
        from runtime_settings import load_runtime_settings

        settings = load_runtime_settings(
            config_data={
                "llm": {
                    "provider": "mistral",
                    "model": "configured-model",
                    "temperature": 0.33,
                    "max_tokens": 333,
                }
            },
            env={},
        )
        response = Mock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": "OK"}}],
        }
        response.raise_for_status.return_value = None

        with (
            patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}, clear=False),
            patch("rag_core.llm.get_runtime_settings", return_value=settings),
            patch("requests.post", return_value=response) as post,
        ):
            answer = ask_mistral_with_context(
                "Donne une réponse neutre.",
                context_text="",
                history=[],
            )

        self.assertEqual(answer, "OK")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "configured-model")
        self.assertEqual(payload["temperature"], 0.33)
        self.assertEqual(payload["max_tokens"], 333)

    def test_compatibility_constants_match_effective_runtime_settings(self):
        from rag_core import constants
        from runtime_settings import get_runtime_settings

        settings = get_runtime_settings()
        retrieval = settings.retrieval

        self.assertEqual(constants.EMBED_MODEL_NAME, retrieval.embedding_model)
        self.assertEqual(constants.RERANK_MODEL_NAME, retrieval.reranker_model)
        self.assertEqual(constants.RERANK_MODEL_FALLBACK, retrieval.reranker_fallback)
        self.assertEqual(constants.RETRIEVE_K, retrieval.retrieve_k)
        self.assertEqual(constants.TOP_K_FAISS, retrieval.top_k_faiss)
        self.assertEqual(constants.TOP_K_BM25, retrieval.top_k_bm25)
        self.assertEqual(constants.HYBRID_ALPHA, retrieval.hybrid_alpha)
        self.assertEqual(constants.EXACT_MATCH_BONUS, retrieval.exact_match_bonus)
        self.assertEqual(constants.MMR_LAMBDA, retrieval.mmr_lambda)
        self.assertEqual(constants.FINAL_K, retrieval.final_k)
        self.assertEqual(constants.MAX_CONTEXT_CHARS, retrieval.max_context_chars)

    def test_benchmark_records_the_same_effective_configuration_and_hash(self):
        from eval.run_current import effective_runtime_configuration
        from runtime_settings import load_runtime_settings

        settings = load_runtime_settings(config_data={}, env={})
        recorded = effective_runtime_configuration(settings, device="cpu")

        self.assertEqual(recorded["runtime"], settings.effective_dict())
        self.assertEqual(recorded["configuration_hash"], settings.config_hash())
        self.assertEqual(
            recorded["runtime"]["retrieval"]["retrieve_k"],
            settings.retrieval.retrieve_k,
        )

    def test_repository_config_matches_the_declared_effective_schema(self):
        from runtime_settings import CONFIG_PATH, load_runtime_settings

        settings = load_runtime_settings(config_path=CONFIG_PATH, env={})
        self.assertTrue(Path(CONFIG_PATH).exists())
        self.assertEqual(settings.retrieval.final_k, 10)
        self.assertEqual(settings.retrieval.max_context_chars, 12000)
        self.assertEqual(settings.generation.model, "mistral-small-latest")
        self.assertEqual(settings.generation.max_tokens, 1200)


if __name__ == "__main__":
    unittest.main()
