import asyncio
import json
import sys
import types
import unittest
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException, Request


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
                "chunk_id": 2,
                "text": "La politique prévoit une conservation de trente jours.",
            },
        )
        self.index = _Index([self.chunk], [0.9])
        self.cache = _Cache()

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
        self.assertEqual(result.sources, [{"path": "C:/docs/policy.txt", "chunk": -1}])
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
        body = AskIn(q="Explique le principe général", source_mode="general")
        with self._patch_pipeline():
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.answer, "Réponse générale contrôlée.")
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
        self.assertEqual(result.sources, [{"path": "C:/docs/policy.txt", "chunk": -1}])
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
        with self._patch_pipeline(llm=llm):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.answer, "No. Tropicalization is no longer possible.")
        self.assertEqual(result.mode, "STRICT(local)")
        self.assertEqual(result.sources, [{"path": "C:/docs/policy.txt", "chunk": -1}])
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
        with self._patch_pipeline(llm=llm):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.answer, "Non. La tropicalisation n'est plus possible.")
        self.assertEqual(result.mode, "STRICT(local)")
        self.assertEqual(result.sources, [{"path": "C:/docs/policy.txt", "chunk": -1}])
        self.assertEqual(result.review.status, "OK")

    def test_fresh_question_with_insufficient_local_context_can_still_ask_web(self):
        def llm(_fn, question, context_text="", **kwargs):
            if str(question).startswith("VERIFIEUR JSON:"):
                return '{"ok":false,"action":"ask_web"}'
            return "Réponse générale sans source locale."

        body = AskIn(q="Quel est le score du match aujourd'hui ?", source_mode="auto")
        with self._patch_pipeline(relevance=False, llm=llm, fresh=True):
            result = run_answer_pipeline(body, _request())

        self.assertEqual(result.answer, "Réponse générale sans source locale.")
        self.assertEqual(result.status, "answered")
        self.assertEqual(result.review.caveat_type, "WEB_RECOMMENDED")
        self.assertTrue(result.review.suggest_web)

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
        self.assertEqual(result.sources, [{"path": "C:/docs/policy.txt", "chunk": -1}])
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
        with self._patch_pipeline(
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

        def pipeline(_body, _request, *, token_sink=None):
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

        def pipeline(_body, _request, *, token_sink=None):
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

        def pipeline(_body, _request, *, token_sink=None):
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

        def pipeline(_body, _request, *, token_sink=None):
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
