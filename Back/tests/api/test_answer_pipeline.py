import asyncio
import json
import re
import sys
import types
import unittest
import uuid
from datetime import datetime, timezone
from contextlib import ExitStack, nullcontext
from enum import Enum
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException, Request
from pydantic import BaseModel


# Les tests unitaires ne doivent ni charger le singleton d'index ni télécharger
# un modèle d'embedding. On expose uniquement le contrat utilisé par le pipeline.
_BACK_ROOT = Path(__file__).resolve().parents[2]
if "api" not in sys.modules:
    api_package = types.ModuleType("api")
    api_package.__path__ = [str(_BACK_ROOT / "api")]
    sys.modules["api"] = api_package

index_singleton_stub = types.ModuleType("api.index_singleton")
index_singleton_stub.idx = object()
index_singleton_stub.format_context_for_llm = Mock()
index_singleton_stub.fuse_contiguous_passages = Mock()
index_singleton_stub.clip_context_blocks = Mock()
index_singleton_stub.ask_mistral_with_context = Mock()
index_singleton_stub.answerability_guard = Mock()
index_singleton_stub.keyword_overlap_count = Mock()
index_singleton_stub.trim_history = lambda rows, max_turns: list(rows)
index_singleton_stub.classify_smalltalk_semantic = Mock(return_value="")
index_singleton_stub.web_search_context = Mock()
index_singleton_stub.RETRIEVE_K = 12
index_singleton_stub.TOP_K_FAISS = 24
index_singleton_stub.HYBRID_ALPHA = 0.65
index_singleton_stub.FUSE_ADJACENT_GAP = 1
index_singleton_stub.MAX_CONTEXT_CHARS = 12000
index_singleton_stub.FINAL_K = 6
sys.modules.setdefault("api.index_singleton", index_singleton_stub)

from api.answer_pipeline import (
    AnswerPipelineResult,
    PostGenerationReview,
    _post_generation_review,
    run_answer_pipeline,
)
from api.schemas import AskIn
from rag_core.faithfulness import (
    AnswerClaim,
    ClaimStatus,
    ClaimVerification,
    FaithfulnessReview,
)
from api.orchestration import OrchestrationPlan, OrchestrationPlanOutputError


class _Cache:
    def __init__(self):
        self.values = {}

    def get(self, key):
        value = self.values.get(key)
        return dict(value) if value else None

    def set(self, key, value):
        self.values[key] = dict(value)


class _Index:
    embed_model = object()

    def __init__(self, results=None, scores=None):
        self.results = results or []
        self.scores = scores or []
        self.search_calls = []

    def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return list(self.results), list(self.scores)


def _request(path="/ask"):
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "query_string": b"",
            "server": ("test", 80),
            "client": ("test", 1),
            "scheme": "http",
        }
    )


class AnswerPipelineTests(unittest.TestCase):
    def setUp(self):
        self.chunk = (
            0.9,
            {
                "idx": 7,
                "path": "C:/docs/policy.txt",
                "source": "file",
                "document_id": "doc_policy",
                "document_metadata": {"origin_path": "C:/docs/policy.txt", "indexed_path": "C:/docs/policy.txt"},
                "chunk_id": 2,
                "text": "La politique prévoit une conservation de trente jours.",
            },
        )
        self.index = _Index([self.chunk], [0.9])
        self.cache = _Cache()

    def assert_local_policy_source(self, sources):
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["document_id"], "doc_policy")
        self.assertEqual(sources[0]["type"], "local_file")

    def test_orchestrator_execution_observability_distinguishes_fast_slow_and_failures(self):
        from api import answer_pipeline as pipeline

        settings = pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"timeout_seconds": 25, "slow_warning_seconds": 10})
        with patch.object(pipeline, "ORCHESTRATOR_SETTINGS", settings):
            fast = pipeline._orchestrator_observability(latency_ms=2_000, failed=False)
            slow = pipeline._orchestrator_observability(latency_ms=12_000, failed=False)
            timeout = pipeline._orchestrator_observability(latency_ms=25_000, failed=True, failure_type="timeout")
            api_error = pipeline._orchestrator_observability(latency_ms=100, failed=True, failure_type="api_error")
            validation_error = pipeline._orchestrator_observability(latency_ms=100, failed=True, failure_type="validation_error")

        self.assertFalse(fast["orchestrator_slow"])
        self.assertFalse(fast["orchestrator_timed_out"])
        self.assertTrue(slow["orchestrator_slow"])
        self.assertFalse(slow["orchestrator_failed"])
        self.assertTrue(timeout["orchestrator_timed_out"])
        self.assertTrue(timeout["orchestrator_failed"])
        self.assertEqual(api_error["orchestrator_failure_type"], "api_error")
        self.assertEqual(validation_error["orchestrator_failure_type"], "validation_error")

    def test_timeout_cancels_queued_orchestrator_work_without_a_retry(self):
        from api import answer_pipeline as pipeline

        future = Mock()
        future.result.side_effect = pipeline.FuturesTimeout()
        executor = Mock()
        executor.submit.return_value = future
        with patch.object(pipeline, "EXEC", executor):
            with self.assertRaises(pipeline.FuturesTimeout):
                pipeline._with_timeout(lambda: None, timeout=25)

        future.cancel.assert_called_once_with()
        executor.submit.assert_called_once()

    def _patch_pipeline(
        self,
        *,
        index=None,
        route_mode="strict_local",
        relevance=True,
        faithfulness=None,
        post_review_enabled=True,
        faithfulness_enabled=True,
        llm=None,
        fresh=False,
        context_text="[1] La politique prévoit une conservation de trente jours.",
    ):
        from api import answer_pipeline as pipeline

        chosen_index = index or self.index
        llm_impl = llm or self._llm
        stack = ExitStack()
        stack.enter_context(patch.object(pipeline, "idx", chosen_index))
        stack.enter_context(patch.object(pipeline, "response_cache", self.cache))
        stack.enter_context(patch.object(pipeline, "_touch_session", Mock()))
        stack.enter_context(patch.object(pipeline, "_get_roleplay", return_value=False))
        stack.enter_context(patch.object(pipeline, "_set_roleplay", Mock()))
        stack.enter_context(patch.object(pipeline, "detect_roleplay_trigger", return_value=None))
        stack.enter_context(patch.object(pipeline, "classify_smalltalk_semantic", return_value=""))
        stack.enter_context(patch.object(pipeline, "_looks_like_equation", return_value=False))
        stack.enter_context(patch.object(pipeline, "_is_explain_followup", return_value=False))
        stack.enter_context(patch.object(pipeline, "_local_index_ready", return_value=True))
        stack.enter_context(patch.object(pipeline, "fuse_contiguous_passages", side_effect=lambda rows, gap: rows))
        stack.enter_context(patch.object(pipeline, "clip_context_blocks", side_effect=lambda rows, **kwargs: rows))
        stack.enter_context(
            patch.object(
                pipeline,
                "format_context_for_llm",
                return_value=context_text,
            )
        )
        stack.enter_context(patch.object(pipeline, "answerability_guard", return_value=True))
        stack.enter_context(patch.object(pipeline, "keyword_overlap_count", return_value=3))
        stack.enter_context(patch.object(pipeline, "_check_context_relevance", return_value=relevance))
        stack.enter_context(patch.object(pipeline, "_llm_route", return_value=route_mode))
        stack.enter_context(patch.object(pipeline, "ENABLE_EXPANSION", False))
        stack.enter_context(patch.object(pipeline, "ENABLE_CONDENSATION", False))
        stack.enter_context(patch.object(pipeline, "ENABLE_POST_GENERATION_REVIEW", post_review_enabled))
        stack.enter_context(patch.object(pipeline, "ENABLE_FAITHFULNESS_CHECK", faithfulness_enabled))
        stack.enter_context(patch.object(pipeline, "_safe_llm", side_effect=llm_impl))
        legacy = faithfulness or {"faithful": True, "score": 0.95, "label": "entailment"}
        status = (
            ClaimStatus.SUPPORTED if legacy["faithful"]
            else ClaimStatus.CONTRADICTED if legacy.get("label") == "contradiction"
            else ClaimStatus.UNSUPPORTED
        )
        claim_review = FaithfulnessReview(
            status="OK" if legacy["faithful"] else "CAVEAT",
            claims=(ClaimVerification(
                claim=AnswerClaim("claim-test", "La conservation est de trente jours.", 0, 38),
                status=status,
                confidence=float(legacy.get("score", 0.0)),
                reason=str(legacy.get("label", "test")),
            ),),
            extraction_ms=0.1,
            verification_ms=0.2,
            total_ms=0.3,
            evidence_chunks_inspected=1,
            model_calls=1,
            model_name="test-nli",
            caveat_required=not legacy["faithful"],
        )
        stack.enter_context(patch.object(pipeline, "verify_answer_claims", return_value=claim_review))
        stack.enter_context(
            patch.object(
                pipeline,
                "should_fallback_to_general",
                side_effect=lambda result, strict_mode: not result["faithful"],
            )
        )
        stack.enter_context(patch.object(pipeline, "_try_get_auth_ids", return_value=(None, None)))
        stack.enter_context(patch.object(pipeline, "_looks_fresh_news", return_value=fresh))
        return stack

    @staticmethod
    def _llm(_fn, question, context_text="", **kwargs):
        if str(question).startswith("VERIFIEUR JSON:"):
            return '{"ok":true,"action":"none"}'
        if context_text:
            return "La conservation est de **trente jours**. <CITATIONS>[1]</CITATIONS>"
        return "Réponse générale contrôlée."

    def test_local_answerable_with_citation_and_validation(self):
        body = AskIn(
            q="Quelle est la durée de conservation de la politique ?",
            source_mode="local",
            thread_id=str(uuid.uuid4()),
        )
        with self._patch_pipeline():
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.answer, "La conservation est de **trente jours**.")
        self.assertEqual(result.mode, "STRICT(local)")
        self.assertEqual(result.status, "answered")
        self.assert_local_policy_source(result.sources)
        self.assertTrue(result.validation_performed)
        self.assertTrue(result.validations["faithfulness"]["faithful"])

    def test_local_reliable_answer_streams_generation_fragments_without_caveat(self):
        from api import answer_pipeline as pipeline

        streamed = []
        body = AskIn(q="Quelle est la durée de conservation ?", source_mode="local")
        with (
            self._patch_pipeline(),
            patch.object(
                pipeline,
                "_safe_llm_stream",
                return_value=iter([
                    "La conservation est de ",
                    "trente jours. <CITATIONS>[1]</CITATIONS>",
                ]),
            ),
        ):
            result = run_answer_pipeline(body, _request(), token_sink=streamed.append)

        self.assertEqual(streamed, [
            "La conservation est de ",
            "trente jours. <CITATIONS>[1]</CITATIONS>",
        ])
        self.assertEqual(result.answer, "La conservation est de trente jours.")
        self.assertEqual(result.review.status, "OK")

    def test_claim_faithfulness_flag_off_skips_checker_and_keeps_streaming(self):
        from api import answer_pipeline as pipeline
        from rag_core import faithfulness as faithfulness_module

        streamed = []
        body = AskIn(q="Quelle est la durée de conservation ?", source_mode="local")
        with (
            self._patch_pipeline(post_review_enabled=False, faithfulness_enabled=False),
            patch.object(
                pipeline,
                "_safe_llm_stream",
                return_value=iter([
                    "La conservation est de ",
                    "trente jours. <CITATIONS>[1]</CITATIONS>",
                ]),
            ),
            patch.object(faithfulness_module, "extract_answer_claims") as extract_claims,
            patch.object(faithfulness_module, "_load_nli_model") as load_nli,
            patch.object(pipeline, "_post_generation_review") as post_review,
        ):
            checker = pipeline.verify_answer_claims
            result = run_answer_pipeline(body, _request(), token_sink=streamed.append)

        checker.assert_not_called()
        extract_claims.assert_not_called()
        load_nli.assert_not_called()
        post_review.assert_not_called()
        self.assertEqual(streamed, [
            "La conservation est de ",
            "trente jours. <CITATIONS>[1]</CITATIONS>",
        ])
        self.assertIsNone(result.faithfulness_review)
        self.assertNotIn("claim_faithfulness", result.validations)
        self.assertNotIn("faithfulness", result.validations)
        self.assertNotIn("post_answer", result.validations)
        self.assertEqual(result.review.status, "OK")
        self.assertFalse(result.review.has_caveat)

    def test_claim_faithfulness_flag_on_still_runs_checker(self):
        from api import answer_pipeline as pipeline

        body = AskIn(q="Quelle est la durée de conservation ?", source_mode="local")
        with self._patch_pipeline(post_review_enabled=True, faithfulness_enabled=True):
            checker = pipeline.verify_answer_claims
            result = run_answer_pipeline(body, _request())

        checker.assert_called_once()
        self.assertIsNotNone(result.faithfulness_review)
        self.assertTrue(result.validations["claim_faithfulness"]["performed"])

    def test_post_generation_review_flag_off_skips_all_caveats(self):
        from api import answer_pipeline as pipeline

        body = AskIn(q="Peut-on tropicaliser un 3725 ?", source_mode="local")
        with (
            self._patch_pipeline(
                post_review_enabled=False,
                faithfulness_enabled=False,
                context_text=(
                    "[1] Document publié en 2021 : la tropicalisation du positionneur "
                    "3730 n'est plus proposée."
                ),
            ),
            patch.object(pipeline, "_post_generation_review") as post_review,
        ):
            result = run_answer_pipeline(body, _request())

        post_review.assert_not_called()
        self.assertIsNone(result.faithfulness_review)
        self.assertEqual(result.review.status, "OK")
        self.assertIsNone(result.review.caveat_type)
        self.assertIsNone(result.review.message)

    def test_post_generation_review_can_be_reenabled_explicitly(self):
        from api import answer_pipeline as pipeline

        expected = PostGenerationReview(
            status="CAVEAT",
            caveat_type="INDIRECT_EVIDENCE",
            message="Réserve de test réactivée.",
            severity="warning",
        )
        body = AskIn(q="Question locale", source_mode="local")
        with (
            self._patch_pipeline(post_review_enabled=True, faithfulness_enabled=False),
            patch.object(pipeline, "_post_generation_review", return_value=expected) as post_review,
        ):
            result = run_answer_pipeline(body, _request())

        post_review.assert_called_once()
        self.assertEqual(result.review, expected)
        self.assertTrue(result.validations["post_answer"]["performed"])

    def test_flag_off_keeps_independent_stale_source_guard(self):
        body = AskIn(q="Cette politique est-elle applicable ?", source_mode="local")
        with self._patch_pipeline(
            faithfulness_enabled=False,
            context_text="[1] Politique publiée en 2021 et toujours applicable à cette date.",
        ):
            result = run_answer_pipeline(body, _request())

        self.assertIsNone(result.faithfulness_review)
        self.assertEqual(result.review.caveat_type, "STALE_SOURCE")

    def test_local_unanswerable_abstains_without_generation(self):
        empty_index = _Index([], [])
        llm = Mock(side_effect=AssertionError("generation must not run"))
        body = AskIn(q="Information absente du corpus", source_mode="local")
        with self._patch_pipeline(
            index=empty_index,
            relevance=False,
            faithfulness_enabled=False,
            llm=llm,
        ):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.mode, "STRICT(local)")
        self.assertEqual(result.sources, [])
        self.assertIsNotNone(result.abstention_reason)

    def test_general_mode_skips_retrieval(self):
        from api import answer_pipeline as pipeline

        body = AskIn(q="Explique le principe général", source_mode="general")
        with self._patch_pipeline(), patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": False})):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.answer, "Réponse générale contrôlée.")
        self.assertEqual(result.mode, "GENERAL(no-context)")
        self.assertEqual(self.index.search_calls, [])

    def test_explicit_general_mode_skips_orchestrator_and_all_retrieval(self):
        from api import answer_pipeline as pipeline

        with (
            self._patch_pipeline(),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", side_effect=AssertionError("orchestrator must not run")),
            patch.object(pipeline, "web_search_context", side_effect=AssertionError("Web must not run")),
        ):
            result = run_answer_pipeline(AskIn(q="Explique le principe general", source_mode="general"), _request())

        self.assertEqual(result.mode, "GENERAL(no-context)")
        self.assertEqual(self.index.search_calls, [])

    def test_auto_mode_uses_shared_routing(self):
        body = AskIn(
            q="Quelle est la durée de conservation de la politique ?",
            source_mode="auto",
        )
        with self._patch_pipeline(route_mode="strict_local"):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.mode, "STRICT(local)")
        self.assertEqual(result.status, "answered")

    def test_condensation_query_is_the_one_retrieved(self):
        from api import answer_pipeline as pipeline

        body = AskIn(
            q="Et sa durée ?",
            source_mode="local",
            history=[
                {"role": "user", "content": "Parle-moi de la politique."},
                {"role": "assistant", "content": "D'accord."},
                {"role": "user", "content": "Quels délais prévoit-elle ?"},
            ],
        )
        with (
            self._patch_pipeline(),
            patch.object(pipeline, "ENABLE_CONDENSATION", True),
            patch.object(
                pipeline,
                "_condense_question",
                return_value="durée de conservation de la politique",
            ) as condense,
        ):
            run_answer_pipeline(body, _request())

        condense.assert_called_once()
        self.assertEqual(
            self.index.search_calls[0][0],
            "durée de conservation de la politique",
        )

    def test_orchestrated_conversation_skips_retrieval(self):
        from api import answer_pipeline as pipeline

        plan = OrchestrationPlan(intent="conversation", needs_retrieval=False, response_strategy="ask_for_missing_information")
        with (
            self._patch_pipeline(),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", return_value=plan),
        ):
            result = run_answer_pipeline(AskIn(q="J'ai reçu une question."), _request())

        self.assertEqual(self.index.search_calls, [])
        self.assertEqual(result.validations["orchestration_needs_retrieval"], False)

    def test_general_question_catalog_probe_promotes_strong_local_title_match(self):
        from api import answer_pipeline as pipeline
        from api.response_trace import ResponseTraceStore

        matching = dict(self.chunk[1])
        matching.update({"document_id": "orion-pdf", "title": "Guide de Project Orion.pdf", "file": "Guide de Project Orion.pdf", "text": "Project Orion is documented locally."})
        index = _Index([(0.9, matching)], [0.9])
        index.corpus = [matching]
        plan = OrchestrationPlan(intent="general_question", needs_retrieval=False, response_strategy="general_answer")
        traces = ResponseTraceStore()
        with (
            self._patch_pipeline(index=index),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", return_value=plan),
            patch.object(pipeline, "RESPONSE_TRACE_ENABLED", True),
            patch.object(pipeline, "RESPONSE_TRACES", traces),
        ):
            result = run_answer_pipeline(AskIn(q="What is Project Orion?"), _request())

        probe = traces.get(result.request_id)["stages"]["catalog_probe"]
        self.assertTrue(probe["strong_match"])
        self.assertGreaterEqual(probe["timing_ms"], 0.0)
        self.assertEqual(index.search_calls[0][0], "What is Project Orion?")

    def test_general_question_catalog_probe_keeps_unmatched_question_general(self):
        from api import answer_pipeline as pipeline
        from api.response_trace import ResponseTraceStore

        index = _Index()
        index.corpus = [{"document_id": "other", "title": "Unrelated financial report.pdf", "file": "Unrelated financial report.pdf"}]
        plan = OrchestrationPlan(intent="general_question", needs_retrieval=False, response_strategy="general_answer")
        traces = ResponseTraceStore()
        with (
            self._patch_pipeline(index=index),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", return_value=plan),
            patch.object(pipeline, "RESPONSE_TRACE_ENABLED", True),
            patch.object(pipeline, "RESPONSE_TRACES", traces),
        ):
            result = run_answer_pipeline(AskIn(q="Explain REST APIs"), _request())

        self.assertTrue(result.mode.startswith("GENERAL"))
        self.assertEqual(index.search_calls, [])
        self.assertFalse(traces.get(result.request_id)["stages"]["catalog_probe"]["strong_match"])

    def test_orchestrated_query_is_retrieved_without_condensation(self):
        from api import answer_pipeline as pipeline

        plan = OrchestrationPlan(
            intent="refine_previous_search", needs_retrieval=True,
            retrieval_query="demande autonome discutée dans les emails",
            use_history=True, source_types=["email"], source_type_provenance={"email": "explicit"}, response_strategy="answer",
        )
        with (
            self._patch_pipeline(),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", return_value=plan),
            patch.object(pipeline, "_condense_question") as condense,
        ):
            run_answer_pipeline(AskIn(q="Regarde les plus récents", source_mode="auto"), _request())

        self.assertTrue(self.index.search_calls[0][0].startswith("Regarde les plus"))
        self.assertIn(plan.retrieval_query, [query for query, _kwargs in self.index.search_calls])
        # The message itself did not select emails, so a planner-proposed
        # source scope cannot become a hard filter.
        self.assertIsNone(self.index.search_calls[0][1]["allowed_sources"])
        condense.assert_not_called()

    def test_enabled_orchestrator_precedes_legacy_semantic_shortcuts_for_documentary_followups(self):
        from api import answer_pipeline as pipeline
        from api.response_trace import ResponseTraceStore

        plans = [
            OrchestrationPlan(intent="document_question", needs_retrieval=True, retrieval_query="status of the request", response_strategy="answer"),
            OrchestrationPlan(intent="refine_previous_search", needs_retrieval=True, retrieval_query="response received on Tuesday about the request", use_history=True, reuse_previous_subject=True, response_strategy="answer"),
            OrchestrationPlan(intent="refine_previous_search", needs_retrieval=True, retrieval_query="verify the status of the request", use_history=True, reuse_previous_subject=True, response_strategy="answer"),
            OrchestrationPlan(intent="refine_previous_search", needs_retrieval=True, retrieval_query="documents concerning the status of the request", use_history=True, reuse_previous_subject=True, response_strategy="answer"),
        ]
        trace_store = ResponseTraceStore()
        messages = [
            "Has my request been accepted?",
            "I think I received an answer on Tuesday.",
            "Please verify it.",
            "Search the documents about it.",
        ]
        history = []
        with (
            self._patch_pipeline(),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "RESPONSE_TRACE_ENABLED", True),
            patch.object(pipeline, "RESPONSE_TRACES", trace_store),
            patch.object(pipeline, "_run_orchestration", side_effect=plans) as orchestrate,
            patch.object(pipeline, "classify_smalltalk_semantic", return_value="insult") as smalltalk,
        ):
            for message in messages:
                result = run_answer_pipeline(AskIn(q=message, history=history), _request())
                trace = trace_store.get(result.request_id)
                self.assertIn("orchestrator_input", trace["stages"])
                self.assertIn("orchestration", trace["stages"])
                self.assertIn("retrieval_input", trace["stages"])
                self.assertIn("retrieval", trace["stages"])
                self.assertNotIn("legacy_bypass", trace["stages"])
                history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": result.answer}])

        self.assertEqual(orchestrate.call_count, len(messages))
        smalltalk.assert_not_called()
        self.assertGreaterEqual(len(self.index.search_calls), len(messages))
        self.assertLessEqual(len(self.index.search_calls), 3 * len(messages))

    def test_invalid_orchestration_falls_back_to_legacy_pipeline(self):
        from api import answer_pipeline as pipeline

        with (
            self._patch_pipeline(),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", side_effect=ValueError("invalid plan")),
        ):
            result = run_answer_pipeline(AskIn(q="Question locale", source_mode="local"), _request())

        self.assertEqual(result.mode, "STRICT(local)")
        self.assertGreaterEqual(len(self.index.search_calls), 1)

    def test_invalid_orchestrator_json_keeps_explicit_email_content_request_in_retrieval(self):
        from api import answer_pipeline as pipeline
        from api.response_trace import ResponseTraceStore

        trace_store = ResponseTraceStore()
        invalid_json = OrchestrationPlanOutputError(
            raw_model_output='{"intent":"document_question","retrieval_query":"unterminated',
            validation_error_details=[{"message": "Unterminated string"}],
        )
        with (
            self._patch_pipeline(),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "RESPONSE_TRACE_ENABLED", True),
            patch.object(pipeline, "RESPONSE_TRACES", trace_store),
            patch.object(pipeline, "_run_orchestration", side_effect=invalid_json),
        ):
            result = run_answer_pipeline(AskIn(q="Pourquoi tu n'étudies pas le contenu du mail ?", source_mode="local"), _request())

        trace = trace_store.get(result.request_id)["stages"]["orchestration"]
        self.assertEqual(len(self.index.search_calls), 1)
        self.assertEqual(trace["error_category"], "model_output_validation_error")
        self.assertEqual(trace["fallback_strategy"], "documentary_retrieval")

    def test_provider_failure_preserves_documentary_retrieval_and_trace_details(self):
        from api import answer_pipeline as pipeline
        from api.response_trace import ResponseTraceStore

        class BadRequestError(Exception):
            code = "unsupported_parameter"
            status_code = 400

        trace_store = ResponseTraceStore()
        with (
            self._patch_pipeline(),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "RESPONSE_TRACE_ENABLED", True),
            patch.object(pipeline, "RESPONSE_TRACES", trace_store),
            patch.object(pipeline, "_run_orchestration", side_effect=BadRequestError("reasoning parameter rejected")),
        ):
            result = run_answer_pipeline(AskIn(q="Was my request finally approved?"), _request())

        trace = trace_store.get(result.request_id)["stages"]["orchestration"]
        self.assertEqual(result.mode, "STRICT(local)")
        self.assertEqual(len(self.index.search_calls), 1)
        self.assertEqual(trace["error_category"], "provider_error")
        self.assertEqual(trace["provider_error_type"], "BadRequestError")
        self.assertEqual(trace["provider_error_code"], "unsupported_parameter")
        self.assertEqual(trace["fallback_strategy"], "documentary_retrieval")

    def test_provider_failure_on_internal_project_approval_uses_retrieval(self):
        from api import answer_pipeline as pipeline

        class BadRequestError(Exception):
            pass

        with (
            self._patch_pipeline(),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", side_effect=BadRequestError("bad request")),
        ):
            result = run_answer_pipeline(AskIn(q="Can you check whether Project Alpha was approved?"), _request())

        self.assertEqual(result.mode, "STRICT(local)")
        self.assertEqual(len(self.index.search_calls), 1)

    def test_provider_failure_keeps_real_conversation_and_general_questions_out_of_forced_retrieval(self):
        from api import answer_pipeline as pipeline

        class BadRequestError(Exception):
            pass

        conversation = type("Turn", (), {"turn_type": "conversational_continuation", "decision_reason": "test"})()
        general = type("Turn", (), {"turn_type": "answer_seeking", "decision_reason": "test"})()
        with (
            self._patch_pipeline(),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", side_effect=BadRequestError("bad request")),
            patch.object(pipeline, "classify_turn", side_effect=[conversation, general]),
            patch.object(pipeline, "_llm_route", return_value="general"),
        ):
            hello = run_answer_pipeline(AskIn(q="Hello"), _request())
            explain = run_answer_pipeline(AskIn(q="Explain REST APIs"), _request())

        self.assertEqual(hello.mode, "GENERAL(conversational)")
        self.assertTrue(explain.mode.startswith("GENERAL"))

    def test_retrieval_parameters_and_order_are_preserved(self):
        from api import answer_pipeline as pipeline

        body = AskIn(q="Durée de conservation ?", source_mode="local")
        with self._patch_pipeline():
            run_answer_pipeline(body, _request())

        self.assertEqual(len(self.index.search_calls), 1)
        _query, kwargs = self.index.search_calls[0]
        self.assertEqual(
            kwargs,
            {
                "retrieve_k": pipeline.RETRIEVE_K,
                "top_k_faiss": pipeline.TOP_K_FAISS,
                "hybrid_alpha": pipeline.HYBRID_ALPHA,
                "use_rerank": True,
                "allowed_sources": {"email", "pdf", "file"},
            },
        )

    def test_expansion_reuses_the_same_retrieval_parameters(self):
        from api import answer_pipeline as pipeline

        expanding_index = _Index()
        expanding_index.search = Mock(
            side_effect=[
                ([], []),
                ([self.chunk], [0.9]),
                ([self.chunk], [0.9]),
            ]
        )
        body = AskIn(q="Formulation éloignée", source_mode="local")
        with (
            self._patch_pipeline(index=expanding_index),
            patch.object(pipeline, "ENABLE_EXPANSION", True),
            patch.object(
                pipeline,
                "_multi_query_expand",
                return_value=["Formulation éloignée", "variante A", "variante B"],
            ),
            patch.object(
                pipeline,
                "_check_context_relevance",
                side_effect=[False, True],
            ),
        ):
            run_answer_pipeline(body, _request())

        self.assertEqual(
            [call.args[0] for call in expanding_index.search.call_args_list],
            ["Formulation éloignée", "variante A", "variante B"],
        )
        kwargs = [call.kwargs for call in expanding_index.search.call_args_list]
        self.assertTrue(all(item == kwargs[0] for item in kwargs))

    def test_web_live_mode_uses_web_context_and_sources(self):
        from api import answer_pipeline as pipeline

        web_context = "[WEB] Source officielle\nhttps://example.test/news\nInformation récente."
        body = AskIn(q="Quelle est l'information récente ?", source_mode="web_live")
        with (
            self._patch_pipeline(),
            patch.object(pipeline, "_with_timeout", return_value=web_context),
        ):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.mode, "STRICT(web_live)")
        self.assertEqual(
            result.sources,
            [{"path": "https://example.test/news", "chunk": -1}],
        )
        self.assertTrue(result.validations["faithfulness"]["faithful"])

    def test_explicit_web_mode_bypasses_orchestrated_local_retrieval(self):
        from api import answer_pipeline as pipeline

        local_plan = OrchestrationPlan(
            intent="document_question", needs_retrieval=True,
            retrieval_query="Los Angeles or San Francisco", response_strategy="answer",
        )
        web_context = "[WEB] Travel source\nhttps://example.test/travel\nComparison."
        with (
            self._patch_pipeline(),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", return_value=local_plan),
            patch.object(pipeline, "_with_timeout", return_value=web_context),
        ):
            result = run_answer_pipeline(AskIn(q="Los Angeles ou San Francisco ?", source_mode="web_live"), _request())

        self.assertEqual(result.mode, "STRICT(web_live)")
        self.assertEqual(self.index.search_calls, [])
        self.assertEqual(result.sources, [{"path": "https://example.test/travel", "chunk": -1}])

    def test_auto_web_followup_uses_the_active_subject_not_source_instruction(self):
        from api import answer_pipeline as pipeline

        plan = OrchestrationPlan(
            intent="web_search", needs_retrieval=True,
            retrieval_query="Los Angeles vs San Francisco for a stay in the United States",
            use_history=True, reuse_previous_subject=True, response_strategy="answer",
        )
        web_context = "[WEB] Travel source\nhttps://example.test/travel\nComparison."
        with (
            self._patch_pipeline(),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", return_value=plan),
            patch.object(pipeline, "_with_timeout", return_value=web_context) as timeout_call,
        ):
            result = run_answer_pipeline(AskIn(
                q="Je veux que tu fasses une recherche web", source_mode="auto",
                history=[
                    {"role": "user", "content": "Tu conseilles Los Angeles ou San Francisco pour un sejour aux States ?"},
                    {"role": "assistant", "content": "Je peux comparer les deux villes."},
                ],
            ), _request())

        self.assertEqual(result.mode, "STRICT(web_live)")
        self.assertEqual(self.index.search_calls, [])
        self.assertIn("Los Angeles vs San Francisco", timeout_call.call_args.args[1])
        self.assertNotEqual(timeout_call.call_args.args[1], "Je veux que tu fasses une recherche web")

    def test_explicit_local_mode_stays_local_when_orchestrator_selects_web(self):
        from api import answer_pipeline as pipeline

        web_plan = OrchestrationPlan(
            intent="web_search", needs_retrieval=True,
            retrieval_query="Los Angeles vs San Francisco", response_strategy="answer",
        )
        with (
            self._patch_pipeline(),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", return_value=web_plan),
            patch.object(pipeline, "web_search_context", side_effect=AssertionError("Web must not run in local mode")),
        ):
            result = run_answer_pipeline(AskIn(q="Los Angeles ou San Francisco ?", source_mode="local"), _request())

        self.assertEqual(result.mode, "STRICT(local)")
        self.assertGreaterEqual(len(self.index.search_calls), 1)

    def test_gefa_followup_keeps_previous_local_source_before_web(self):
        from api import answer_pipeline as pipeline
        from api.source_planner import SourcePlanItem

        price_chunk = (0.99, {
            **self.chunk[1],
            "file": "TARIF GEFA 2025.pdf",
            "path": "C:/docs/TARIF GEFA 2025.pdf",
            "text": "GEFA KG2 DN50 purchase price HT 2023: 54.21 EUR.",
        })
        index = _Index([price_chunk], [0.99])
        plan = OrchestrationPlan(
            intent="refine_previous_search", needs_retrieval=True,
            retrieval_query="GEFA KG2 DN50 price", use_history=True,
            reuse_previous_subject=True, response_strategy="answer",
            source_plan=[
                SourcePlanItem(source="web", priority=1),
                SourcePlanItem(source="local", priority=2, required=True),
            ],
        )
        with (
            self._patch_pipeline(
                index=index,
                context_text="[1] GEFA KG2 DN50 purchase price HT 2023: 54.21 EUR.",
                post_review_enabled=False, faithfulness_enabled=False,
            ),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", return_value=plan),
            patch.object(pipeline, "web_search_context", side_effect=AssertionError("Web must not run before active local source")),
        ):
            result = run_answer_pipeline(AskIn(
                q="Tu pourrais me donner le prix d'une KG2 DN50 ?", source_mode="auto",
                history=[
                    {"role": "user", "content": "Tu as acces a la price list GEFA ?"},
                    {"role": "assistant", "content": "Oui.", "meta": {
                        "generation_mode": "STRICT(local)",
                        "sources": [{"path": "C:/docs/TARIF GEFA 2025.pdf", "document_id": "doc_policy"}],
                    }},
                ],
            ), _request())

        self.assertNotEqual(result.route_mode, "web_live")
        self.assertEqual(result.validations["source_plan"][0]["source"], "local")
        self.assertEqual(result.validations["active_source_context"]["source"], "local")
        self.assertGreaterEqual(len(index.search_calls), 1)

    def test_blocking_orchestrator_clarification_skips_retrieval(self):
        from api import answer_pipeline as pipeline
        from api.source_planner import SourcePlanItem

        plan = OrchestrationPlan(
            intent="document_question", needs_retrieval=True,
            retrieval_query="GEFA KG2 DN50 price", response_strategy="ask_for_missing_information",
            source_plan=[SourcePlanItem(source="local", priority=1, required=True)],
            clarification_needed=True,
            clarification_reason="purchase_and_sale_prices_differ",
            missing_information=["price_type"],
            clarification_question="Tu veux le prix d'achat ou le prix de vente France ?",
            ambiguity_level="blocking",
        )
        with (
            self._patch_pipeline(post_review_enabled=False, faithfulness_enabled=False),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", return_value=plan),
            patch.object(pipeline, "web_search_context", side_effect=AssertionError("No source should run before clarification")),
        ):
            result = run_answer_pipeline(AskIn(q="Quel est le prix de la KG2 DN50 ?", source_mode="auto"), _request())

        self.assertEqual(result.route_mode, "clarification")
        self.assertEqual(result.validations["next_source_action"], "ASK_CLARIFICATION")
        self.assertEqual(self.index.search_calls, [])

    def test_local_fact_can_use_a_separate_general_complement(self):
        from api import answer_pipeline as pipeline
        from api.source_planner import SourcePlanItem

        seen = {}
        def llm(_fn, _question, context_text="", **kwargs):
            if context_text:
                seen.update(kwargs)
                return "La conservation dure trente jours. Explication generale. <CITATIONS>[1]</CITATIONS>"
            return "Explication generale."

        plan = OrchestrationPlan(
            intent="document_question", needs_retrieval=True,
            retrieval_query="duree conservation", response_strategy="answer",
            source_plan=[
                SourcePlanItem(source="local", priority=1, required=True),
                SourcePlanItem(source="general", priority=2, complementary=True),
            ],
        )
        with (
            self._patch_pipeline(post_review_enabled=False, faithfulness_enabled=False, llm=llm),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", return_value=plan),
        ):
            result = run_answer_pipeline(AskIn(
                q="Quelle est la duree de conservation et a quoi sert cette regle ?",
                source_mode="auto",
            ), _request())

        self.assertEqual(result.mode, "STRICT(multi-source+general)")
        self.assertTrue(seen["allow_general_complement"])
        self.assertEqual(result.validations["sources_checked"], ["local", "general"])
        self.assertEqual(result.validations["claim_sources"][-1]["source_type"], "general")
        self.assertFalse(result.validations["claim_sources"][-1]["citation_required"])

    def test_required_web_complement_runs_after_local_product_evidence(self):
        from api import answer_pipeline as pipeline
        from api.source_planner import SourcePlanItem

        local_chunk = (0.99, {
            **self.chunk[1],
            "text": "The 82.7 HV02 product is certified NACE MR0175.",
        })
        index = _Index([local_chunk], [0.99])
        plan = OrchestrationPlan(
            intent="document_question", needs_retrieval=True,
            retrieval_query="82.7 HV02 NACE certification", response_strategy="answer",
            source_plan=[
                SourcePlanItem(source="local", priority=1, required=True),
                SourcePlanItem(source="web", priority=2, required=True),
            ],
        )
        web = Mock(return_value="[WEB] NACE standard\nhttps://example.test/nace\nNACE MR0175 remains current.")
        with (
            self._patch_pipeline(
                index=index,
                context_text="[1] The 82.7 HV02 product is certified NACE MR0175.",
                post_review_enabled=False, faithfulness_enabled=False,
            ),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", return_value=plan),
            patch.object(pipeline, "web_search_context", web),
        ):
            result = run_answer_pipeline(AskIn(
                q="Le 82.7 HV02 est-il NACE et cette certification est-elle encore valable ?",
                source_mode="auto",
            ), _request())

        self.assertEqual(result.mode, "STRICT(multi-source)")
        self.assertEqual(result.validations["sources_checked"], ["local", "web"])
        self.assertTrue(result.validations["multi_source_used"])
        self.assertEqual(web.call_count, 1)

    def test_faithfulness_review_preserves_draft_and_adds_inference_caveat(self):
        calls = {"strict": 0}

        def llm(_fn, question, context_text="", **kwargs):
            if str(question).startswith("VERIFIEUR JSON:"):
                return '{"ok":true,"action":"none"}'
            if context_text:
                calls["strict"] += 1
                return "Brouillon non fidèle. <CITATIONS>[1]</CITATIONS>"
            return "Réponse de fallback validée."

        body = AskIn(q="Question avec fallback", source_mode="local")
        with self._patch_pipeline(
            faithfulness={"faithful": False, "score": 0.1, "label": "contradiction"},
            llm=llm,
        ):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(calls["strict"], 1)
        self.assertEqual(result.answer, "Brouillon non fidèle.")
        self.assertEqual(result.mode, "STRICT(local)")
        self.assertEqual(result.status, "answered")
        self.assert_local_policy_source(result.sources)
        self.assertEqual(result.review.caveat_type, "CONTRADICTED_CLAIM")

    def test_unsupported_claim_uses_existing_post_generation_caveat_channel(self):
        body = AskIn(q="Question locale", source_mode="local")
        with self._patch_pipeline(
            faithfulness={"faithful": False, "score": 0.8, "label": "neutral"},
        ):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.answer, "La conservation est de **trente jours**.")
        self.assertEqual(result.review.caveat_type, "UNSUPPORTED_CLAIM")
        self.assertIsNotNone(result.faithfulness_review)
        self.assertEqual(
            result.faithfulness_review["status_counts"]["UNSUPPORTED"], 1
        )

    def test_wrong_claim_citation_maps_to_one_existing_partial_evidence_caveat(self):
        from api import answer_pipeline as pipeline

        claim_review = FaithfulnessReview(
            status="CAVEAT",
            claims=(ClaimVerification(
                claim=AnswerClaim("claim-citation", "Le 3730 possède un PCB verni.", 0, 31),
                status=ClaimStatus.SUPPORTED,
                confidence=0.95,
                reason="Supported by another context block",
                citation_correct=False,
            ),),
            extraction_ms=0.1,
            verification_ms=0.2,
            total_ms=0.3,
            evidence_chunks_inspected=2,
            model_calls=1,
            model_name="test-nli",
            caveat_required=True,
        )
        with patch.object(pipeline, "_safe_llm", side_effect=AssertionError("no duplicate review call")):
            review = _post_generation_review(
                "Le PCB est-il verni ?",
                "Le 3730 possède un PCB verni.",
                "Contexte",
                [],
                claim_review=claim_review,
            )

        self.assertEqual(review.caveat_type, "CITATION_MISMATCH")
        self.assertIn("source citée", review.message or "")

    def test_english_tropicalization_keeps_sourced_local_answer_when_verifier_asks_web(self):
        def llm(_fn, question, context_text="", **kwargs):
            if str(question).startswith("VERIFIEUR JSON:"):
                return '{"ok":false,"action":"ask_web"}'
            return "No. Tropicalization is no longer possible. <CITATIONS>[1]</CITATIONS>"

        body = AskIn(
            q="Is it possible to tropicalize a 3730 positionner?",
            source_mode="local",
        )
        evidence = (0.9, {**self.chunk[1], "text": "The 3730 tropicalization is no longer possible."})
        with self._patch_pipeline(index=_Index([evidence], [0.9]), llm=llm):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.answer, "No. Tropicalization is no longer possible.")
        self.assertEqual(result.mode, "STRICT(local)")
        self.assert_local_policy_source(result.sources)
        self.assertEqual(result.review.caveat_type, "WEB_RECOMMENDED")
        self.assertTrue(result.review.suggest_web)

    def test_french_tropicalization_keeps_existing_local_behavior(self):
        def llm(_fn, question, context_text="", **kwargs):
            if str(question).startswith("VERIFIEUR JSON:"):
                return '{"ok":true,"action":"none"}'
            return "Non. La tropicalisation n'est plus possible. <CITATIONS>[1]</CITATIONS>"

        body = AskIn(
            q="C'est possible de tropicaliser un 3730 ?",
            source_mode="local",
        )
        evidence = (0.9, {**self.chunk[1], "text": "La tropicalisation du 3730 n'est plus possible."})
        with self._patch_pipeline(index=_Index([evidence], [0.9]), llm=llm):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.answer, "Non. La tropicalisation n'est plus possible.")
        self.assertEqual(result.mode, "STRICT(local)")
        self.assert_local_policy_source(result.sources)
        self.assertEqual(result.review.status, "OK")

    def test_fresh_question_uses_web_directly(self):
        def llm(_fn, question, context_text="", **kwargs):
            if str(question).startswith("VERIFIEUR JSON:"):
                return '{"ok":false,"action":"ask_web"}'
            return "Réponse générale sans source locale."

        body = AskIn(q="Quel est le score du match aujourd'hui ?", source_mode="auto")
        with (
            self._patch_pipeline(relevance=False, llm=llm, fresh=True),
            patch("api.answer_pipeline.web_search_context", return_value="[WEB] Match source\nhttps://example.test/match\nCurrent score."),
        ):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.mode, "STRICT(web_live)")
        self.assertEqual(result.sources, [{"path": "https://example.test/match", "chunk": -1}])

    def test_strict_answer_without_returnable_source_can_still_ask_web(self):
        sourceless_chunk = (
            0.9,
            {
                "idx": 8,
                "path": "",
                "chunk_id": 3,
                "text": "Contexte strict sans chemin de source exploitable.",
            },
        )

        def llm(_fn, question, context_text="", **kwargs):
            if str(question).startswith("VERIFIEUR JSON:"):
                return '{"ok":false,"action":"ask_web"}'
            return "Brouillon strict sans source retournable. <CITATIONS>[1]</CITATIONS>"

        body = AskIn(q="Question stable sans source retournable", source_mode="local")
        with self._patch_pipeline(
            index=_Index([sourceless_chunk], [0.9]),
            llm=llm,
        ):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.answer, "Brouillon strict sans source retournable.")
        self.assertEqual(result.sources, [])
        self.assertEqual(result.review.caveat_type, "WEB_RECOMMENDED")

    def test_sourced_local_answer_ignores_spurious_ask_web_action(self):
        def llm(_fn, question, context_text="", **kwargs):
            if str(question).startswith("VERIFIEUR JSON:"):
                return '{"ok":false,"action":"ask_web"}'
            return "Réponse locale validée. <CITATIONS>[1]</CITATIONS>"

        body = AskIn(q="Question locale stable", source_mode="local")
        with self._patch_pipeline(llm=llm):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.answer, "Réponse locale validée.")
        self.assert_local_policy_source(result.sources)
        self.assertEqual(result.review.caveat_type, "WEB_RECOMMENDED")

    def test_post_verifier_none_action_remains_unchanged(self):
        def llm(_fn, question, context_text="", **kwargs):
            if str(question).startswith("VERIFIEUR JSON:"):
                return '{"ok":true,"action":"none"}'
            return "Réponse locale inchangée. <CITATIONS>[1]</CITATIONS>"

        body = AskIn(q="Question locale stable", source_mode="local")
        with self._patch_pipeline(llm=llm):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.answer, "Réponse locale inchangée.")
        self.assertEqual(result.review.status, "OK")

    def test_old_source_adds_specific_stale_source_caveat(self):
        body = AskIn(q="Cette politique est-elle applicable ?", source_mode="local")
        with self._patch_pipeline(
            context_text="[1] Politique publiée en 2021 et toujours applicable à cette date.",
        ):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.answer, "La conservation est de **trente jours**.")
        self.assertEqual(result.review.caveat_type, "STALE_SOURCE")
        self.assertIn("2021", result.review.message or "")
        self.assertTrue(result.review.suggest_web)

    def test_nearby_product_only_adds_indirect_evidence_caveat(self):
        body = AskIn(q="Peut-on tropicaliser un 3725 ?", source_mode="local")
        evidence = (0.9, {**self.chunk[1], "text": "La tropicalisation du positionneur 3730 n'est plus proposée."})
        with self._patch_pipeline(
            index=_Index([evidence], [0.9]),
            faithfulness_enabled=False,
            context_text="[1] La tropicalisation du positionneur 3730 n'est plus proposée.",
        ):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.review.caveat_type, "INDIRECT_EVIDENCE")
        self.assertIn("3730", result.review.message or "")
        self.assertIn("3725", result.review.message or "")

    def test_conflicting_sources_add_typed_caveat(self):
        def llm(_fn, question, context_text="", **kwargs):
            if str(question).startswith("VERIFIEUR JSON:"):
                return json.dumps({
                    "status": "CAVEAT",
                    "caveat_type": "CONFLICTING_EVIDENCE",
                    "message": "Une source autorise l'opération, tandis qu'une autre l'interdit.",
                    "severity": "warning",
                    "suggest_web": False,
                })
            return "Les documents ne concordent pas. <CITATIONS>[1]</CITATIONS>"

        body = AskIn(q="Cette opération est-elle autorisée ?", source_mode="local")
        with self._patch_pipeline(llm=llm):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.answer, "Les documents ne concordent pas.")
        self.assertEqual(result.review.caveat_type, "CONFLICTING_EVIDENCE")
        self.assertIn("interdit", result.review.message or "")

    def test_conversational_introductions_skip_condensation_and_retrieval(self):
        from api import answer_pipeline as pipeline

        formulations = [
            "J'ai rencontré une difficulté avec un appareil.",
            "On vient de me soumettre un sujet.",
            "Je vais commencer par vous donner le contexte.",
        ]
        for question in formulations:
            index = _Index([self.chunk], [0.9])
            with (
                self._patch_pipeline(index=index),
                patch.object(pipeline, "classify_turn", return_value=type("Turn", (), {"turn_type": "conversational_continuation", "decision_reason": "test"})()),
            ):
                result = run_answer_pipeline(AskIn(q=question), _request())
            self.assertEqual(index.search_calls, [])
            self.assertEqual(result.mode, "GENERAL(conversational)")
            self.assertEqual(result.route_mode, "general")

    def test_answer_seeking_turns_still_retrieve_without_question_mark_rule(self):
        from api import answer_pipeline as pipeline

        formulations = [
            "Peut-on adapter le module AX-17 ?",
            "Est-il possible d'adapter le module AX-17 ?",
            "On me demande si le module AX-17 peut être adapté.",
            "Que puis-je répondre concernant l'adaptation du module AX-17 ?",
            "Je cherche des informations sur la possibilité d'adapter le module AX-17.",
        ]
        for question in formulations:
            index = _Index([self.chunk], [0.9])
            with (
                self._patch_pipeline(index=index),
                patch.object(pipeline, "classify_turn", return_value=type("Turn", (), {"turn_type": "answer_seeking", "decision_reason": "test"})()),
            ):
                result = run_answer_pipeline(AskIn(q=question), _request())
            self.assertEqual(len(index.search_calls), 1)
            self.assertEqual(result.mode, "STRICT(local)")

    def test_evidence_modes_are_direct_related_or_none_without_domain_rules(self):
        from api.answer_pipeline import _classify_evidence_mode

        self.assertEqual(
            _classify_evidence_mode("Le module AX-17 est-il compatible ?", "Le module AX-17 est compatible.", relevant=True),
            "direct",
        )
        self.assertEqual(
            _classify_evidence_mode("Le module AX-17 est-il compatible ?", "Le module AX-18 est compatible.", relevant=True),
            "related",
        )
        self.assertEqual(
            _classify_evidence_mode("Le module AX-17 est-il compatible ?", "Une procédure administrative générale.", relevant=False),
            "none",
        )
        self.assertEqual(
            _classify_evidence_mode(
                "Le module AX-17 est-il compatible ?",
                "Le module AX-18 est compatible. Le module AX-19 ne l'est pas.",
                relevant=True,
            ),
            "related",
        )

    def test_related_evidence_reaches_generator_with_a_reservation_policy(self):
        from api import answer_pipeline as pipeline

        observed = {}

        def llm(_fn, _question, context_text="", **kwargs):
            observed.update(kwargs)
            return "Le cas AX-18 est documenté, mais il ne permet pas de conclure pour AX-17. <CITATIONS>[1]</CITATIONS>"

        related = (0.9, {**self.chunk[1], "text": "Le module AX-18 est compatible."})
        with (
            self._patch_pipeline(index=_Index([related], [0.9]), llm=llm, context_text="[1] Le module AX-18 est compatible."),
            patch.object(pipeline, "classify_turn", return_value=type("Turn", (), {"turn_type": "answer_seeking", "decision_reason": "test"})()),
        ):
            result = run_answer_pipeline(AskIn(q="Le module AX-17 est-il compatible ?", source_mode="local"), _request())

        self.assertEqual(observed["evidence_mode"], "related")
        self.assertTrue(observed["exact_entity_guard"])
        self.assertEqual(observed["missing_exact_entities"], ["ax-17"])
        self.assertIn("AX-18", result.answer)

    def test_related_evidence_bypasses_lexical_relevance_abstention_and_keeps_sources(self):
        from api import answer_pipeline as pipeline

        related = (0.91, {**self.chunk[1], "file": "related-reference.txt", "text": "La référence AX-18 est compatible."})
        with (
            self._patch_pipeline(
                index=_Index([related], [0.91]), relevance=False,
                context_text="[1] La référence AX-18 est compatible.",
                post_review_enabled=False, faithfulness_enabled=False,
            ),
            patch.object(pipeline, "classify_turn", return_value=type("Turn", (), {"turn_type": "answer_seeking", "decision_reason": "test"})()),
        ):
            result = run_answer_pipeline(AskIn(q="La référence AX-17 est-elle compatible ?", source_mode="local"), _request())

        self.assertEqual(result.answer, "La conservation est de **trente jours**.")
        self.assert_local_policy_source(result.sources)
        self.assertEqual(result.sources[0]["display_name"], "related-reference.txt")
        self.assertNotEqual(result.status, "abstained")

    def test_auto_keeps_related_local_evidence_instead_of_falling_back_to_web(self):
        from api import answer_pipeline as pipeline

        related = (0.91, {**self.chunk[1], "text": "La tropicalisation du positionneur ZX-730 est documentee."})
        with (
            self._patch_pipeline(
                index=_Index([related], [0.91]),
                context_text="[1] La tropicalisation du positionneur ZX-730 est documentee.",
                post_review_enabled=False, faithfulness_enabled=False,
            ),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": False})),
            patch.object(pipeline, "_looks_fresh_news", return_value=False),
            patch.object(pipeline, "_llm_route", side_effect=AssertionError("related local evidence must stay local")),
            patch.object(pipeline, "web_search_context", side_effect=AssertionError("Web must not run")),
        ):
            result = run_answer_pipeline(AskIn(q="Peut-on tropicaliser le positionneur ZX-725 ?", source_mode="auto"), _request())

        self.assertEqual(result.mode, "STRICT(local)")
        self.assertEqual(result.route_mode, "strict_local")
        self.assert_local_policy_source(result.sources)

    def test_auto_can_fallback_to_web_only_when_local_context_is_irrelevant(self):
        from api import answer_pipeline as pipeline

        unrelated = (0.91, {**self.chunk[1], "text": "Le calendrier des conges et la facturation sont disponibles."})
        web_context = "[WEB] Product source\nhttps://example.test/product\nTechnical information."
        with (
            self._patch_pipeline(
                index=_Index([unrelated], [0.91]), relevance=False,
                context_text="[1] Le calendrier des conges et la facturation sont disponibles.",
                route_mode="web_live", post_review_enabled=False, faithfulness_enabled=False,
            ),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": False})),
            patch.object(pipeline, "_looks_fresh_news", return_value=False),
            patch.object(pipeline, "web_search_context", return_value=web_context),
        ):
            result = run_answer_pipeline(AskIn(q="Peut-on tropicaliser le positionneur ZX-725 ?", source_mode="auto"), _request())

        self.assertEqual(result.mode, "STRICT(web_live)")
        self.assertEqual(result.route_mode, "web_live")
        self.assertEqual(result.sources, [{"path": "https://example.test/product", "chunk": -1}])

    def test_auto_sends_current_public_information_to_web_before_local_retrieval(self):
        from api import answer_pipeline as pipeline

        web_context = "[WEB] Fare source\nhttps://example.test/fare\nCurrent fare."
        with (
            self._patch_pipeline(post_review_enabled=False, faithfulness_enabled=False),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": False})),
            patch.object(pipeline, "_looks_fresh_news", return_value=True),
            patch.object(pipeline, "web_search_context", return_value=web_context),
        ):
            result = run_answer_pipeline(AskIn(q="Quel est le prix actuel d'un billet entre deux villes ?", source_mode="auto"), _request())

        self.assertEqual(result.mode, "STRICT(web_live)")
        self.assertEqual(self.index.search_calls, [])

    def test_auto_keeps_incomplete_internal_decision_context_local(self):
        from api import answer_pipeline as pipeline

        partial = (0.91, {**self.chunk[1], "text": "Le projet Orion a ete examine, sans decision finale consignée."})
        with (
            self._patch_pipeline(
                index=_Index([partial], [0.91]),
                context_text="[1] Le projet Orion a ete examine, sans decision finale consignée.",
                post_review_enabled=False, faithfulness_enabled=False,
            ),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": False})),
            patch.object(pipeline, "_looks_fresh_news", return_value=False),
            patch.object(pipeline, "_llm_route", side_effect=AssertionError("partial internal evidence must stay local")),
            patch.object(pipeline, "web_search_context", side_effect=AssertionError("Web must not run")),
        ):
            result = run_answer_pipeline(AskIn(q="Quelle decision avons-nous prise sur le projet Orion ?", source_mode="auto"), _request())

        self.assertEqual(result.mode, "STRICT(local)")
        self.assertEqual(result.route_mode, "strict_local")

    def test_auto_uses_general_for_irrelevant_non_web_question(self):
        from api import answer_pipeline as pipeline

        unrelated = (0.91, {**self.chunk[1], "text": "Le calendrier des conges est disponible."})
        with (
            self._patch_pipeline(
                index=_Index([unrelated], [0.91]), relevance=False, route_mode="general",
                context_text="[1] Le calendrier des conges est disponible.",
                post_review_enabled=False, faithfulness_enabled=False,
            ),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": False})),
            patch.object(pipeline, "_looks_fresh_news", return_value=False),
            patch.object(pipeline, "web_search_context", side_effect=AssertionError("Web must not run")),
        ):
            result = run_answer_pipeline(AskIn(q="Explique la difference entre deux concepts abstraits.", source_mode="auto"), _request())

        self.assertTrue(result.mode.startswith("GENERAL"))
        self.assertEqual(result.route_mode, "general")

    def test_orchestrator_web_plan_keeps_active_subject_in_web_query(self):
        from api import answer_pipeline as pipeline

        plan = OrchestrationPlan(
            intent="web_search", needs_retrieval=True,
            retrieval_query="flights from Lyon to Los Angeles over the next six months",
            use_history=True, reuse_previous_subject=True, response_strategy="answer",
        )
        web = Mock(return_value="[WEB] Flight source\nhttps://example.test/flights\nAvailable flights.")
        with (
            self._patch_pipeline(post_review_enabled=False, faithfulness_enabled=False),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", return_value=plan),
            patch.object(pipeline, "web_search_context", web),
        ):
            result = run_answer_pipeline(AskIn(
                q="Tu peux me trouver un vol pour les prochains mois depuis Lyon ?", source_mode="auto",
                history=[
                    {"role": "user", "content": "Los Angeles ou San Francisco pour un sejour ?"},
                    {"role": "assistant", "content": "Los Angeles est recommande."},
                ],
            ), _request())

        self.assertEqual(result.mode, "STRICT(web_live)")
        self.assertEqual(self.index.search_calls, [])
        self.assertIn("Los Angeles", web.call_args.args[0])
        self.assertIn("Lyon", web.call_args.args[0])

    def test_orchestrator_failure_recovers_active_subject_before_web_fallback(self):
        from api import answer_pipeline as pipeline

        indeed = (0.91, {**self.chunk[1], "text": "Offre d'emploi Indeed pour ingenieur commercial a Lyon."})
        index = _Index([indeed], [0.91])
        failed_plan = OrchestrationPlanOutputError(raw_model_output="", validation_error_details=[{"message": "empty output"}])
        web = Mock(return_value="[WEB] Flight source\nhttps://example.test/flights\nAvailable flights.")
        with (
            self._patch_pipeline(
                index=index, relevance=False, route_mode="web_live",
                context_text="[1] Offre d'emploi Indeed pour ingenieur commercial a Lyon.",
                post_review_enabled=False, faithfulness_enabled=False,
            ),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", side_effect=failed_plan),
            patch.object(pipeline, "_looks_fresh_news", return_value=False),
            patch.object(pipeline, "classify_turn", return_value=type("Turn", (), {"turn_type": "answer_seeking", "decision_reason": "test"})()),
            patch.object(pipeline, "web_search_context", web),
        ):
            result = run_answer_pipeline(AskIn(
                q="Tu peux me trouver un vol pour les prochains mois depuis Lyon ?", source_mode="auto",
                history=[
                    {"role": "user", "content": "Je pars a Los Angeles cet ete."},
                    {"role": "assistant", "content": "Destination retenue."},
                ],
            ), _request())

        self.assertEqual(result.mode, "STRICT(web_live)")
        self.assertIn("Los Angeles", index.search_calls[0][0])
        self.assertIn("Los Angeles", web.call_args.args[0])

    def test_lone_shared_location_does_not_make_job_offers_relevant_to_travel(self):
        from api.answer_pipeline import _evaluate_evidence, _classify_local_context_state

        jobs = [(0.91, {"text": "Offre d'emploi Indeed pour ingenieur commercial a Lyon."})]
        decision = _evaluate_evidence(
            "Trouve-moi un vol depuis Lyon", jobs,
            guard_ok=True, context_is_relevant=False, overlap=1,
        )
        self.assertEqual(decision.mode, "none")
        self.assertEqual(decision.context_relevance_reason, "information_need_misaligned")
        self.assertEqual(_classify_local_context_state(decision), "irrelevant")

    def test_related_product_reference_remains_relevant_but_incomplete_in_auto(self):
        from api.answer_pipeline import _evaluate_evidence, _classify_local_context_state

        technical = [(0.91, {"text": "Le Trovis 3730 peut etre tropicalise avec vernis et limites techniques."})]
        decision = _evaluate_evidence(
            "Peut-on tropicaliser un Trovis 3725 ?", technical,
            guard_ok=True, context_is_relevant=True, overlap=1,
        )
        self.assertEqual(decision.mode, "related")
        self.assertEqual(_classify_local_context_state(decision), "relevant_but_incomplete")

    def test_partial_internal_project_context_remains_local_in_auto(self):
        from api import answer_pipeline as pipeline

        partial = (0.91, {**self.chunk[1], "text": "Le projet Alpha a ete examine, sans decision finale consignée."})
        with (
            self._patch_pipeline(
                index=_Index([partial], [0.91]),
                context_text="[1] Le projet Alpha a ete examine, sans decision finale consignée.",
                post_review_enabled=False, faithfulness_enabled=False,
            ),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": False})),
            patch.object(pipeline, "_looks_fresh_news", return_value=False),
            patch.object(pipeline, "_llm_route", side_effect=AssertionError("partial project context must remain local")),
            patch.object(pipeline, "web_search_context", side_effect=AssertionError("Web must not run")),
        ):
            result = run_answer_pipeline(AskIn(q="Quelle decision a ete prise sur le projet Alpha ?", source_mode="auto"), _request())

        self.assertEqual(result.mode, "STRICT(local)")

    def test_local_mode_allows_web_only_for_explicit_web_request_plan(self):
        from api import answer_pipeline as pipeline

        plan = OrchestrationPlan(
            intent="web_search", needs_retrieval=True, web_request_explicit=True,
            retrieval_query="public information about the requested subject", response_strategy="answer",
        )
        with (
            self._patch_pipeline(post_review_enabled=False, faithfulness_enabled=False),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", return_value=plan),
            patch.object(pipeline, "web_search_context", return_value="[WEB] Source\nhttps://example.test/source\nPublic result."),
        ):
            result = run_answer_pipeline(AskIn(q="Regarde sur le web ce sujet", source_mode="local"), _request())

        self.assertEqual(result.mode, "STRICT(web_live)")
        self.assertEqual(self.index.search_calls, [])

    def test_explicit_web_request_in_auto_uses_web_when_orchestrator_succeeds(self):
        from api import answer_pipeline as pipeline

        plan = OrchestrationPlan(
            intent="web_search", needs_retrieval=True, web_request_explicit=True,
            retrieval_query="flights from Lyon to Los Angeles in the next six months",
            use_history=True, reuse_previous_subject=True, response_strategy="answer",
        )
        with (
            self._patch_pipeline(post_review_enabled=False, faithfulness_enabled=False),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", return_value=plan),
            patch.object(pipeline, "web_search_context", return_value="[WEB] Source\nhttps://example.test/flights\nFlights."),
        ):
            result = run_answer_pipeline(AskIn(q="Fais une recherche web plutot", source_mode="auto"), _request())

        self.assertEqual(result.mode, "STRICT(web_live)")
        self.assertEqual(self.index.search_calls, [])

    def test_explicit_web_request_survives_orchestrator_timeout_with_followup_subject(self):
        from api import answer_pipeline as pipeline

        web = Mock(return_value="[WEB] Source\nhttps://example.test/flights\nFlights.")
        with (
            self._patch_pipeline(post_review_enabled=False, faithfulness_enabled=False),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", side_effect=TimeoutError("timeout")),
            patch.object(pipeline, "_condense_question", return_value="Recherche web de vols Lyon vers Los Angeles pour les six prochains mois."),
            patch.object(pipeline, "web_search_context", web),
        ):
            result = run_answer_pipeline(AskIn(
                q="Non mais fait une recherche web plutot", source_mode="auto",
                history=[{"role": "user", "content": "Je cherche un vol Lyon vers Los Angeles dans les six prochains mois."}],
            ), _request())

        self.assertEqual(result.mode, "STRICT(web_live)")
        self.assertEqual(self.index.search_calls, [])
        self.assertIn("Lyon vers Los Angeles", web.call_args.args[0])

    def test_explicit_web_request_survives_invalid_orchestrator_output(self):
        from api import answer_pipeline as pipeline

        invalid = OrchestrationPlanOutputError(raw_model_output="", validation_error_details=[{"message": "empty output"}])
        with (
            self._patch_pipeline(post_review_enabled=False, faithfulness_enabled=False),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", side_effect=invalid),
            patch.object(pipeline, "web_search_context", return_value="[WEB] Source\nhttps://example.test/source\nResult."),
        ):
            result = run_answer_pipeline(AskIn(q="Cherche sur internet ce sujet", source_mode="auto"), _request())

        self.assertEqual(result.mode, "STRICT(web_live)")
        self.assertEqual(self.index.search_calls, [])

    def test_orchestrator_timeout_without_explicit_web_request_does_not_force_web(self):
        from api import answer_pipeline as pipeline

        unrelated = (0.91, {**self.chunk[1], "text": "Le calendrier des conges est disponible."})
        with (
            self._patch_pipeline(
                index=_Index([unrelated], [0.91]), relevance=False, route_mode="general",
                context_text="[1] Le calendrier des conges est disponible.",
                post_review_enabled=False, faithfulness_enabled=False,
            ),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", side_effect=TimeoutError("timeout")),
            patch.object(pipeline, "_looks_fresh_news", return_value=False),
            patch.object(pipeline, "_llm_route", return_value="general"),
            patch.object(pipeline, "classify_turn", return_value=type("Turn", (), {"turn_type": "answer_seeking", "decision_reason": "test"})()),
            patch.object(pipeline, "web_search_context", side_effect=AssertionError("Web must not run")),
        ):
            result = run_answer_pipeline(AskIn(q="Peux-tu resumer ceci ?", source_mode="auto"), _request())

        self.assertTrue(result.mode.startswith("GENERAL"))

    def test_partial_and_chronological_evidence_reach_generation_with_sources(self):
        from api import answer_pipeline as pipeline

        observed = {}

        def llm(_fn, _question, context_text="", **kwargs):
            observed["context"] = context_text
            observed.update(kwargs)
            return (
                "Les sources décrivent une première étape, puis l'absence de décision finale. "
                "Je ne peux donc pas confirmer la conclusion demandée. <CITATIONS>[1,2]</CITATIONS>"
            )

        partial = [
            (0.93, {**self.chunk[1], "text": "Le 4 avril, la demande AX-17 a été déposée."}),
            (0.91, {**self.chunk[1], "text": "Le 12 avril, aucune décision finale concernant AX-17 n'était encore reçue."}),
        ]
        with (
            self._patch_pipeline(
                index=_Index(partial, [0.93, 0.91]),
                llm=llm,
                context_text="[1] Le 4 avril, la demande AX-17 a été déposée.\n[2] Le 12 avril, aucune décision finale concernant AX-17 n'était encore reçue.",
                post_review_enabled=False,
                faithfulness_enabled=False,
            ),
            patch.object(pipeline, "classify_turn", return_value=type("Turn", (), {"turn_type": "answer_seeking", "decision_reason": "test"})()),
        ):
            result = run_answer_pipeline(AskIn(q="La demande AX-17 a-t-elle été acceptée ?", source_mode="local"), _request())

        self.assertEqual(observed["evidence_mode"], "direct")
        self.assertIn("première étape", result.answer)
        self.assertIn("ne peux donc pas confirmer", result.answer)
        self.assert_local_policy_source(result.sources)
        self.assertNotEqual(result.status, "abstained")

    def test_contradictory_direct_evidence_reaches_generation_with_sources(self):
        from api import answer_pipeline as pipeline

        def llm(_fn, _question, context_text="", **kwargs):
            return "Les sources divergent ; elles ne permettent pas d'arbitrer. <CITATIONS>[1,2]</CITATIONS>"

        contradictory = [
            (0.93, {**self.chunk[1], "text": "La référence AX-17 est approuvée."}),
            (0.92, {**self.chunk[1], "text": "La référence AX-17 n'est pas approuvée."}),
        ]
        with (
            self._patch_pipeline(
                index=_Index(contradictory, [0.93, 0.92]),
                llm=llm,
                context_text="[1] La référence AX-17 est approuvée.\n[2] La référence AX-17 n'est pas approuvée.",
                post_review_enabled=False,
                faithfulness_enabled=False,
            ),
            patch.object(pipeline, "classify_turn", return_value=type("Turn", (), {"turn_type": "answer_seeking", "decision_reason": "test"})()),
        ):
            result = run_answer_pipeline(AskIn(q="La référence AX-17 est-elle approuvée ?", source_mode="local"), _request())

        self.assertIn("divergent", result.answer)
        self.assert_local_policy_source(result.sources)
        self.assertNotEqual(result.status, "abstained")

    def test_vaguely_similar_block_remains_none_and_abstains(self):
        from api.answer_pipeline import _evaluate_evidence

        block = (0.8, {"text": "Un sujet général sans relation exploitable.", "path": "C:/docs/general.txt"})
        decision = _evaluate_evidence(
            "La référence AX-17 est-elle compatible ?", [block],
            guard_ok=False, context_is_relevant=False, overlap=0,
        )
        self.assertEqual(decision.mode, "none")
        self.assertEqual(decision.reason, "guard_rejected_final_blocks")

    def test_evidence_decision_handles_direct_related_and_explicit_comparison(self):
        from api.answer_pipeline import _evaluate_evidence

        direct = [(0.9, {"text": "La référence AX-17 est compatible."})]
        related = [
            (0.9, {"text": "La référence AX-18 est compatible."}),
            (0.85, {"text": "La version AX-19 est compatible dans les mêmes conditions."}),
        ]
        comparison = [(0.9, {"text": "AX-17 diffère de AX-18 sur cette propriété."})]

        self.assertEqual(_evaluate_evidence("AX-17 est-il compatible ?", direct, guard_ok=True, context_is_relevant=True, overlap=2).mode, "direct")
        self.assertEqual(_evaluate_evidence("AX-17 est-il compatible ?", related, guard_ok=True, context_is_relevant=False, overlap=1).mode, "related")
        self.assertEqual(_evaluate_evidence("Comparer AX-17 et AX-18", comparison, guard_ok=True, context_is_relevant=True, overlap=2).mode, "direct")

    def test_related_evidence_accepts_morphological_reformulations_without_domain_rules(self):
        from api.answer_pipeline import _evaluate_evidence

        related = [(0.9, {"text": "La tropicalisation du produit AX-18 est document\u00e9e."})]
        decision = _evaluate_evidence(
            "Peut-on tropicaliser le produit AX-17 ?",
            related,
            guard_ok=True,
            context_is_relevant=False,
            overlap=0,
        )

        self.assertEqual(decision.mode, "related")
        self.assertEqual(decision.reason, "different_anchor_with_guard_and_topic_relation")

    def test_comparison_with_both_requested_entities_is_direct(self):
        from api.answer_pipeline import _classify_evidence_mode

        self.assertEqual(
            _classify_evidence_mode(
                "Comparer AX-17 et AX-18", "AX-17 diffère de AX-18 sur cette propriété.", relevant=True
            ),
            "direct",
        )

    def test_persistence_is_executed_once_by_the_shared_pipeline(self):
        from api import answer_pipeline as pipeline

        body = AskIn(
            q="Durée de conservation ?",
            source_mode="local",
            thread_id="chat-1",
        )
        with (
            self._patch_pipeline(),
            patch.object(
                pipeline,
                "_try_get_auth_ids",
                return_value=("tenant-1", "user-1"),
            ),
            patch.object(pipeline, "create_chat") as create,
            patch.object(pipeline, "append_message") as append,
        ):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.chat_id, "chat-1")
        create.assert_called_once()
        self.assertEqual(append.call_count, 2)

    def test_smalltalk_persists_user_and_assistant_without_retrieval(self):
        from api import answer_pipeline as pipeline

        index = _Index([self.chunk], [0.9])
        with (
            self._patch_pipeline(index=index),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": False})),
            patch.object(pipeline, "classify_smalltalk_semantic", return_value="greeting"),
            patch.object(pipeline, "_try_get_auth_ids", return_value=("tenant-1", "user-1")),
            patch.object(pipeline, "create_chat") as create,
            patch.object(pipeline, "append_message") as append,
        ):
            result = run_answer_pipeline(AskIn(q="Bonjour", thread_id="chat-smalltalk"), _request())

        self.assertEqual(index.search_calls, [])
        create.assert_called_once()
        self.assertEqual(append.call_count, 2)
        self.assertEqual([call.args[3] for call in append.call_args_list], ["user", "assistant"])
        self.assertEqual([call.args[4] for call in append.call_args_list], ["Bonjour", result.answer])
        self.assertEqual(result.chat_id, "chat-smalltalk")

    def test_conversational_and_general_paths_persist_without_rag(self):
        from api import answer_pipeline as pipeline

        for body, turn_type in (
            (AskIn(q="Je souhaite expliquer une situation", thread_id="chat-conversation"), "conversational_continuation"),
            (AskIn(q="Explique ce principe", source_mode="general", thread_id="chat-general"), None),
        ):
            index = _Index([self.chunk], [0.9])
            with (
                self._patch_pipeline(index=index),
                patch.object(pipeline, "classify_turn", return_value=type("Turn", (), {"turn_type": turn_type, "decision_reason": "test"})()) if turn_type else nullcontext(),
                patch.object(pipeline, "_try_get_auth_ids", return_value=("tenant-1", "user-1")),
                patch.object(pipeline, "create_chat"),
                patch.object(pipeline, "append_message") as append,
            ):
                run_answer_pipeline(body, _request())
            self.assertEqual(index.search_calls, [])
            self.assertEqual([call.args[3] for call in append.call_args_list], ["user", "assistant"])

    def test_streamed_turn_persists_one_complete_assistant_message(self):
        from api import answer_pipeline as pipeline

        streamed = []
        with (
            self._patch_pipeline(post_review_enabled=False, faithfulness_enabled=False),
            patch.object(pipeline, "_try_get_auth_ids", return_value=("tenant-1", "user-1")),
            patch.object(pipeline, "create_chat"),
            patch.object(pipeline, "append_message") as append,
            patch.object(pipeline, "_safe_llm_stream", return_value=iter([
                "Première partie ", "et dernière partie. <CITATIONS>[1]</CITATIONS>",
            ])),
        ):
            result = run_answer_pipeline(
                AskIn(q="Question locale", source_mode="local", thread_id="chat-stream"),
                _request("/ask/stream"),
                token_sink=streamed.append,
            )

        self.assertEqual(streamed, ["Première partie ", "et dernière partie. <CITATIONS>[1]</CITATIONS>"])
        self.assertEqual(append.call_count, 2)
        self.assertEqual(append.call_args_list[1].args[3], "assistant")
        self.assertEqual(append.call_args_list[1].args[4], result.answer)

    def test_assistant_persistence_failure_is_logged_and_result_survives(self):
        from api import answer_pipeline as pipeline

        streamed = []
        def append_with_failure(*args, **kwargs):
            if args[3] == "assistant":
                raise RuntimeError("sqlite unavailable")

        with (
            self._patch_pipeline(post_review_enabled=False, faithfulness_enabled=False),
            patch.object(pipeline, "_try_get_auth_ids", return_value=("tenant-1", "user-1")),
            patch.object(pipeline, "create_chat"),
            patch.object(pipeline, "append_message", side_effect=append_with_failure),
            patch.object(pipeline, "_safe_llm_stream", return_value=iter(["Réponse complète"])),
            patch("builtins.print") as printed,
        ):
            result = run_answer_pipeline(
                AskIn(q="Question locale", source_mode="local", thread_id="chat-persist"),
                _request("/ask/stream"),
                token_sink=streamed.append,
            )

        self.assertEqual(result.answer, "Réponse complète")
        logs = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn('"event": "stream_error"', logs)
        self.assertIn('"stage": "message_persistence"', logs)
        self.assertIn("sqlite unavailable", logs)


    def test_processing_error_keeps_the_already_persisted_user_turn(self):
        from api import answer_pipeline as pipeline

        with (
            self._patch_pipeline(),
            patch.object(pipeline, "classify_smalltalk_semantic", return_value="greeting"),
            patch.object(pipeline, "_try_get_auth_ids", return_value=("tenant-1", "user-1")),
            patch.object(pipeline, "create_chat"),
            patch.object(pipeline, "append_message") as append,
            patch.object(pipeline, "_safe_llm", side_effect=RuntimeError("generation failed")),
        ):
            with self.assertRaises(HTTPException):
                run_answer_pipeline(AskIn(q="Bonjour", thread_id="chat-error"), _request())

        self.assertEqual(append.call_count, 1)
        self.assertEqual(append.call_args.args[3], "user")

    def test_cached_response_keeps_the_same_public_result(self):
        llm = Mock(side_effect=self._llm)
        body = AskIn(q="Durée mise en cache ?", source_mode="local")
        with self._patch_pipeline(llm=llm):
            first = run_answer_pipeline(body, _request())
            second = run_answer_pipeline(body, _request())

        self.assertEqual(first.answer, second.answer)
        self.assertEqual(first.sources, second.sources)
        self.assertEqual(first.mode, second.mode)
        self.assertEqual(first.context_length, second.context_length)
        self.assertEqual(len(self.index.search_calls), 2)
        self.assertEqual(llm.call_count, 2)
        self.assertFalse(second.validation_performed)

    def test_response_cache_never_crosses_conversations_and_keys_the_used_history(self):
        from api import answer_pipeline as pipeline

        generated = Mock(side_effect=["answer-a", "answer-b", "answer-c", "history-one", "history-two"])
        with (
            self._patch_pipeline(),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": False})),
            patch.object(pipeline, "_generate_answer", side_effect=generated),
        ):
            # Identical question and RAG context in three chats: no cross-chat reuse.
            answers = [
                run_answer_pipeline(AskIn(q="Question identique", source_mode="local", thread_id=chat_id), _request()).answer
                for chat_id in ("chat-a", "chat-b", "chat-c")
            ]
            self.assertEqual(answers, ["answer-a", "answer-b", "answer-c"])
            self.assertEqual(generated.call_count, 3)

            # The same chat can reuse only the exact history/context scope.
            first = run_answer_pipeline(AskIn(
                q="Question avec historique", source_mode="local", thread_id="chat-history",
                history=[{"role": "user", "content": "Contexte A"}],
            ), _request())
            changed_history = run_answer_pipeline(AskIn(
                q="Question avec historique", source_mode="local", thread_id="chat-history",
                history=[{"role": "user", "content": "Contexte B"}],
            ), _request())
            exact_repeat = run_answer_pipeline(AskIn(
                q="Question avec historique", source_mode="local", thread_id="chat-history",
                history=[{"role": "user", "content": "Contexte A"}],
            ), _request())

        self.assertEqual((first.answer, changed_history.answer, exact_repeat.answer), ("history-one", "history-two", "history-one"))
        self.assertEqual(generated.call_count, 5)


    @staticmethod
    def _supported_history(answer="La bande morte se règle avec PSCS et le kit de paramétrage. La plage est de 0,5 à 5 %. La procédure détaillée n’est pas disponible."):
        return [
            {"role": "user", "content": "Comment régler la bande morte des actionneurs PS AMS ?"},
            {"role": "assistant", "content": answer, "meta": {
                "message_id": "assistant-1",
                "sources": [{"path": "20260701_Confirmation Dead Band PS-AMS_signed_PSA.pdf", "chunk": 2}],
                "evidence_provenance": {"evidence_sufficient": True, "evidence_chunk_uids": ["ps-ams:2"]},
            }},
        ]

    def test_email_transformation_reuses_previous_supported_answer_without_retrieval(self):
        from api import answer_pipeline as pipeline

        generated = json.dumps({"answer": "Voici un brouillon :", "artifacts": [{"type": "email_draft", "subject": "Réglage de la bande morte PS-AMS", "content": "Bonjour,\n\nLa bande morte se règle avec PSCS et le kit de paramétrage. La plage est de 0,5 à 5 %.\n\nCordialement,"}], "citations": [1]})
        with (
            self._patch_pipeline(),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", return_value=OrchestrationPlan(intent="document_question", needs_retrieval=True, retrieval_query="PS AMS dead band", response_strategy="answer")),
            patch.object(pipeline, "_generate_answer", return_value=generated) as generate,
        ):
            result = run_answer_pipeline(AskIn(q="Je réponds quoi par mail ?", history=self._supported_history()), _request())

        self.assertEqual(self.index.search_calls, [])
        self.assertEqual(result.mode, "GROUNDED_TRANSFORMATION")
        self.assertEqual(result.validations["response_format"], "email_draft")
        self.assertTrue(result.validations["reuse_previous_answer"])
        self.assertTrue(result.validations["previous_evidence_reused"])
        self.assertEqual(result.sources[0]["path"], "20260701_Confirmation Dead Band PS-AMS_signed_PSA.pdf")
        self.assertTrue(generate.call_args.kwargs["grounded_transformation"])
        self.assertIn("PREVIOUS_SUPPORTED_ANSWER", generate.call_args.args[1])

    def test_invalid_orchestrator_json_recovers_grounded_transformation(self):
        from api import answer_pipeline as pipeline

        invalid = OrchestrationPlanOutputError(raw_model_output='{"intent":"document_question",', validation_error_details=[{"message": "Unterminated string"}])
        with (
            self._patch_pipeline(),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", side_effect=invalid),
            patch.object(pipeline, "_generate_answer", return_value="La bande morte se règle avec PSCS et le kit de paramétrage. <CITATIONS>[1]</CITATIONS>"),
        ):
            result = run_answer_pipeline(AskIn(q="Fais-moi un mail avec ça", history=self._supported_history()), _request())

        self.assertEqual(self.index.search_calls, [])
        self.assertTrue(result.validations["grounded_transformation"])
        self.assertTrue(result.validations["orchestrator_failure_recovery"])
        self.assertEqual(result.validations["orchestrator_failure_recovery_reason"], "previous_supported_answer_and_evidence")

    def test_transformation_without_previous_evidence_requests_content_without_retrieval(self):
        from api import answer_pipeline as pipeline

        with self._patch_pipeline(), patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})):
            result = run_answer_pipeline(AskIn(q="Fais-moi un mail avec ça", history=[{"role": "assistant", "content": "Une ancienne réponse sans provenance."}]), _request())

        self.assertEqual(self.index.search_calls, [])
        self.assertEqual(result.mode, "CLARIFICATION")
        self.assertFalse(result.validations["reuse_previous_answer"])

    def test_detail_and_verification_requests_are_not_transformations(self):
        from api import answer_pipeline as pipeline
        from api.answer_pipeline import _detect_response_transformation

        self.assertEqual(_detect_response_transformation("On me demande par mail cette information. Je réponds quoi ?"), "email_draft")
        self.assertEqual(_detect_response_transformation("Fais-en une réponse client"), "email_draft")
        self.assertEqual(_detect_response_transformation("Rédige-moi ça proprement"), "rewrite")
        self.assertIsNone(_detect_response_transformation("Peux-tu me donner la procédure détaillée étape par étape ?"))
        self.assertIsNone(_detect_response_transformation("Vérifie si c’est toujours valable aujourd’hui"))
        plan = OrchestrationPlan(intent="refine_previous_search", needs_retrieval=True, retrieval_query="PS AMS dead band detailed procedure", use_history=True, reuse_previous_subject=True, response_strategy="answer")
        with (
            self._patch_pipeline(),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", return_value=plan),
        ):
            run_answer_pipeline(AskIn(q="Peux-tu me donner la procédure détaillée étape par étape ?", history=self._supported_history()), _request())
        self.assertGreaterEqual(len(self.index.search_calls), 1)


    def test_new_factual_email_draft_without_previous_answer_uses_retrieval(self):
        from api import answer_pipeline as pipeline

        plan = OrchestrationPlan(
            intent="document_question", needs_retrieval=True, retrieval_query="product approval status",
            response_strategy="answer", response_format="email_draft",
        )
        with (
            self._patch_pipeline(post_review_enabled=False, faithfulness_enabled=False),
            patch.object(pipeline, "ORCHESTRATOR_SETTINGS", pipeline.ORCHESTRATOR_SETTINGS.model_copy(update={"enabled": True})),
            patch.object(pipeline, "_run_orchestration", return_value=plan),
        ):
            result = run_answer_pipeline(AskIn(q="A client asks if the product is approved. What should I reply by email?"), _request())

        self.assertGreaterEqual(len(self.index.search_calls), 1)
        self.assertNotEqual(result.mode, "CLARIFICATION")


class TransportParityTests(unittest.TestCase):
    @staticmethod
    async def _collect(response):
        chunks = []
        async for item in response.body_iterator:
            chunks.append(item.decode() if isinstance(item, bytes) else item)
        return "".join(chunks)

    def test_sse_with_post_checks_off_emits_content_then_done_without_caveat(self):
        from api import routes_ask

        result = AnswerPipelineResult(
            answer="Réponse streamée sans post-vérification.",
            review=PostGenerationReview(),
            faithfulness_review=None,
        )

        def pipeline(_body, _request, *, token_sink=None, status_sink=None):
            token_sink("Réponse streamée ")
            token_sink("sans post-vérification.")
            return result

        with patch.object(routes_ask, "run_answer_pipeline", side_effect=pipeline):
            response = asyncio.run(routes_ask.ask_stream(AskIn(q="Question"), _request("/ask/stream")))
            stream = asyncio.run(self._collect(response))

        events = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith("data: ")]
        event_types = [event["type"] for event in events]
        self.assertGreaterEqual(event_types.count("content"), 2)
        self.assertEqual(event_types[-1], "done")
        self.assertTrue(all(event_type == "content" for event_type in event_types[:-1]))
        self.assertNotIn("event: caveat", stream)
        self.assertIsNone(events[-1]["faithfulness_review"])
        self.assertIsNone(events[-1]["review"]["caveat_type"])

    def test_sse_done_uses_canonical_two_source_citations_after_ten_chunk_generation(self):
        """Regression: the client must receive the remapped answer, not the draft."""
        from api import answer_pipeline as pipeline
        from api import routes_ask

        def chunk(index, document_id):
            return (
                0.9,
                {
                    "path": f"C:/docs/{document_id}.txt",
                    "source": "file",
                    "file": f"{document_id}.txt",
                    "document_id": document_id,
                    "document_metadata": {
                        "origin_path": f"C:/docs/{document_id}.txt",
                        "indexed_path": f"C:/docs/{document_id}.txt",
                    },
                    "chunk_id": index,
                    "text": f"Information du document {document_id}, chunk {index}.",
                },
            )

        blocks = [chunk(index, "doc_a" if index in {1, 4} else "doc_b") for index in range(1, 11)]
        raw_answer = "Information A. [1]Information B. [4, 10]<CITATIONS>[1,4,10]</CITATIONS>"
        helper = AnswerPipelineTests()
        helper.setUp()
        with (
            helper._patch_pipeline(
                index=_Index(blocks, [0.9] * 10),
                post_review_enabled=False,
                faithfulness_enabled=False,
                context_text="\n".join(f"[{i}] chunk" for i in range(1, 11)),
            ),
            patch.object(pipeline, "_safe_llm_stream", return_value=iter([raw_answer])),
        ):
            response = asyncio.run(routes_ask.ask_stream(AskIn(q="Question technique", source_mode="local"), _request("/ask/stream")))
            stream = asyncio.run(self._collect(response))

        events = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith("data: ")]
        done = next(event for event in events if event["type"] == "done")
        self.assertEqual(len(done["sources"]), 2)
        self.assertEqual(done["answer"], "Information A. [1]\n\nInformation B. [1, 2]")
        self.assertNotIn("[4", done["answer"])
        self.assertNotIn("10]", done["answer"])
        self.assertNotIn("[1]Information", done["answer"])
        visible_markers = " ".join(re.findall(r"\[[^\]]+\]", done["answer"]))
        visible = [int(value) for value in re.findall(r"\d+", visible_markers)]
        self.assertTrue(all(value <= len(done["sources"]) for value in visible))

    def test_sse_emits_public_pipeline_status_before_streamed_content(self):
        from api import routes_ask

        result = AnswerPipelineResult(answer="Réponse.", review=PostGenerationReview())

        def pipeline(_body, _request, *, token_sink=None, status_sink=None):
            for stage, label in [
                ("analyze_request", "Analyse de la demande"),
                ("identify_response_mode", "Identification du mode de réponse"),
                ("search_documents", "Recherche dans vos documents"),
                ("analyze_results", "Analyse des résultats trouvés"),
                ("prepare_response", "Préparation de la réponse"),
            ]:
                status_sink(stage, label)
            token_sink("Bonjour")
            return result

        with patch.object(routes_ask, "run_answer_pipeline", side_effect=pipeline):
            response = asyncio.run(routes_ask.ask_stream(AskIn(q="Question"), _request("/ask/stream")))
            stream = asyncio.run(self._collect(response))

        frames = [frame for frame in stream.split("\n\n") if frame]
        status_frames = [frame for frame in frames if "event: status" in frame]
        content_index = next(index for index, frame in enumerate(frames) if '"type": "content"' in frame)
        self.assertEqual(len(status_frames), 5)
        self.assertLess(max(frames.index(frame) for frame in status_frames), content_index)
        self.assertIn('"stage": "prepare_response"', status_frames[-1])

    def test_json_and_sse_reconstruct_the_same_validated_answer(self):
        from api import routes_ask

        result = AnswerPipelineResult(
            answer="Réponse finale validée.",
            sources=[{"path": "C:/docs/policy.txt", "chunk": -1}],
            mode="STRICT(local)",
            status="answered",
            context_length=42,
            request_id="request-1",
            validations={"faithfulness": {"faithful": True}},
            review=PostGenerationReview(
                status="CAVEAT",
                caveat_type="PARTIAL_EVIDENCE",
                message="Les sources couvrent uniquement la première partie.",
                severity="warning",
            ),
        )
        body = AskIn(q="Question", source_mode="local")
        with patch.object(routes_ask, "run_answer_pipeline", return_value=result) as shared:
            json_response = routes_ask.ask(body, _request("/ask"))
            stream_response = asyncio.run(
                routes_ask.ask_stream(body, _request("/ask/stream"))
            )
            stream = asyncio.run(self._collect(stream_response))

        events = [
            json.loads(line[6:])
            for line in stream.splitlines()
            if line.startswith("data: ")
        ]
        reconstructed = "".join(
            event.get("content", "") for event in events if event["type"] == "content"
        )
        done = next(event for event in events if event["type"] == "done")

        self.assertEqual(reconstructed, json_response.answer)
        self.assertEqual(done["sources"], json_response.sources)
        self.assertEqual(done["mode"], json_response.mode)
        self.assertEqual(done["status"], result.status)
        self.assertEqual(json_response.review.caveat_type, "PARTIAL_EVIDENCE")
        self.assertEqual(done["review"]["caveat_type"], "PARTIAL_EVIDENCE")
        self.assertIn("event: caveat", stream)
        self.assertEqual(shared.call_count, 2)

    def test_sse_exception_after_response_start_is_safe_and_logged(self):
        from api import routes_ask

        def pipeline(_body, _request, *, token_sink=None, status_sink=None):
            token_sink("Avant erreur")
            raise RuntimeError("secret backend detail")

        with patch.object(routes_ask, "run_answer_pipeline", side_effect=pipeline), patch("builtins.print") as printed:
            response = asyncio.run(routes_ask.ask_stream(AskIn(q="Question"), _request("/ask/stream")))
            stream = asyncio.run(self._collect(response))

        self.assertEqual(response.status_code, 200)
        self.assertIn('"type": "error"', stream)
        self.assertIn('"request_id": "', stream)
        self.assertIn("Une erreur interne est survenue", stream)
        self.assertNotIn("secret backend detail", stream)
        logs = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn('"event": "stream_error"', logs)
        self.assertIn('"error_type": "RuntimeError"', logs)
        self.assertIn("secret backend detail", logs)

    def test_sse_done_serialization_failure_is_logged_as_sse_emit(self):
        from api import routes_ask

        result = AnswerPipelineResult(answer="Réponse", request_id="result-id", review=PostGenerationReview())
        original_sse = routes_ask._sse

        def failing_sse(payload, event=None):
            if payload.get("type") == "done":
                raise ValueError("done emit failed")
            return original_sse(payload, event)

        with patch.object(routes_ask, "run_answer_pipeline", return_value=result), patch.object(routes_ask, "_sse", side_effect=failing_sse), patch("builtins.print") as printed:
            response = asyncio.run(routes_ask.ask_stream(AskIn(q="Question"), _request("/ask/stream")))
            stream = asyncio.run(self._collect(response))

        self.assertIn('"type": "error"', stream)
        logs = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn('"stage": "sse_emit"', logs)
        self.assertIn("done emit failed", logs)

    def test_json_safe_supports_standard_payload_types(self):
        from api.diagnostics import json_safe

        class Kind(Enum):
            PDF = "pdf"

        class Metadata(BaseModel):
            name: str

        value = json_safe({
            "path": Path("C:/docs/policy.pdf"),
            "when": datetime(2026, 8, 28, tzinfo=timezone.utc),
            "kind": Kind.PDF,
            "model": Metadata(name="policy"),
            "set": {"a", "b"},
            "tuple": (1, 2),
        })
        json.dumps(value)
        self.assertEqual(Path(value["path"]), Path("C:/docs/policy.pdf"))
        self.assertEqual(value["kind"], "pdf")
        self.assertEqual(value["model"], {"name": "policy"})

    def test_sse_emits_generation_fragments_before_post_generation_caveat(self):
        from api import routes_ask

        result = AnswerPipelineResult(
            answer="Réponse locale fiable, immédiatement diffusée.",
            sources=[{"path": "C:/docs/policy.txt", "chunk": -1}],
            mode="STRICT(local)",
            review=PostGenerationReview(
                status="CAVEAT",
                caveat_type="WEB_RECOMMENDED",
                message="La source date de 2021 ; une vérification web peut confirmer l'état actuel.",
                severity="warning",
                suggest_web=True,
            ),
        )

        def pipeline(_body, _request, *, token_sink=None, status_sink=None):
            self.assertIsNotNone(token_sink)
            token_sink("Réponse locale fiable, ")
            token_sink("immédiatement diffusée.")
            return result

        with patch.object(routes_ask, "run_answer_pipeline", side_effect=pipeline):
            response = asyncio.run(routes_ask.ask_stream(AskIn(q="Question"), _request("/ask/stream")))
            stream = asyncio.run(self._collect(response))

        frames = [frame for frame in stream.split("\n\n") if frame]
        content_positions = [i for i, frame in enumerate(frames) if '"type": "content"' in frame]
        caveat_position = next(i for i, frame in enumerate(frames) if "event: caveat" in frame)
        done_position = next(i for i, frame in enumerate(frames) if '"type": "done"' in frame)
        self.assertGreaterEqual(len(content_positions), 2)
        self.assertLess(max(content_positions), caveat_position)
        self.assertLess(caveat_position, done_position)
        events = [
            json.loads(line[6:])
            for line in stream.splitlines()
            if line.startswith("data: ")
        ]
        reconstructed = "".join(
            event.get("content", "") for event in events if event["type"] == "content"
        )
        self.assertEqual(reconstructed, result.answer)

    def test_sse_reliable_answer_has_no_caveat_event(self):
        from api import routes_ask

        result = AnswerPipelineResult(answer="Réponse fiable.", review=PostGenerationReview())

        def pipeline(_body, _request, *, token_sink=None, status_sink=None):
            token_sink("Réponse ")
            token_sink("fiable.")
            return result

        with patch.object(routes_ask, "run_answer_pipeline", side_effect=pipeline):
            response = asyncio.run(routes_ask.ask_stream(AskIn(q="Question"), _request("/ask/stream")))
            stream = asyncio.run(self._collect(response))

        self.assertNotIn("event: caveat", stream)
        self.assertIn('"type": "done"', stream)

    def test_sse_upstream_abstention_does_not_invent_generation_or_caveat(self):
        from api import routes_ask

        result = AnswerPipelineResult(
            answer="Je n'ai rien trouvé de pertinent dans les sources autorisées.",
            status="abstained",
            abstention_reason="contexte insuffisant",
            review=PostGenerationReview(),
        )

        def pipeline(_body, _request, *, token_sink=None, status_sink=None):
            # Aucun appel au sink : les contrôles amont ont arrêté le pipeline.
            return result

        with patch.object(routes_ask, "run_answer_pipeline", side_effect=pipeline):
            response = asyncio.run(routes_ask.ask_stream(AskIn(q="Question"), _request("/ask/stream")))
            stream = asyncio.run(self._collect(response))

        events = [json.loads(line[6:]) for line in stream.splitlines() if line.startswith("data: ")]
        contents = [event for event in events if event["type"] == "content"]
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0]["content"], result.answer)
        self.assertNotIn("event: caveat", stream)
        self.assertEqual(events[-1]["status"], "abstained")

    def test_json_and_sse_keep_the_same_empty_question_http_error(self):
        from api import routes_ask

        body = AskIn(q="   ")
        with self.assertRaisesRegex(HTTPException, "Champ 'q' vide"):
            routes_ask.ask(body, _request("/ask"))
        with self.assertRaisesRegex(HTTPException, "Champ 'q' vide"):
            asyncio.run(routes_ask.ask_stream(body, _request("/ask/stream")))


if __name__ == "__main__":
    unittest.main()
