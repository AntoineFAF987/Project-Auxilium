import sys
import types
from pathlib import Path

_BACK_ROOT = Path(__file__).resolve().parents[2]
if str(_BACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACK_ROOT))
if "api" not in sys.modules:
    package = types.ModuleType("api")
    package.__path__ = [str(_BACK_ROOT / "api")]
    sys.modules["api"] = package

from api.multi_query_retrieval import ResolvedUserTask, build_retrieval_queries, build_retrieval_queries_for_task


def test_orchestrated_factual_subject_keeps_original_wording_and_shared_subject():
    """Presentation wording cannot erase either user's retrieval evidence."""
    factual = build_retrieval_queries(
        original_query="Does product X support Y?",
        orchestrator_query="product X support Y",
    )
    email = build_retrieval_queries(
        original_query="A client asks by email whether product X supports Y; what should I reply?",
        orchestrator_query="product X support Y",
    )
    assert factual[0] == ("original_autonomous", "Does product X support Y?")
    assert email[0] == ("original_autonomous", "A client asks by email whether product X supports Y; what should I reply?")
    assert ("orchestrator", "product X support Y") in factual
    assert ("orchestrator", "product X support Y") in email


def assert_paraphrase_factual_invariance(left: ResolvedUserTask, right: ResolvedUserTask, *, left_target_document_ids: list[str], right_target_document_ids: list[str]) -> None:
    """Architecture assertion: presentation cannot change factual retrieval inputs."""
    assert left.exact_entities == right.exact_entities
    assert left.information_need == right.information_need
    assert left.query_semantics == right.query_semantics
    assert left.source_constraints == right.source_constraints
    assert left.temporal_constraints == right.temporal_constraints
    assert build_retrieval_queries_for_task(left).queries == build_retrieval_queries_for_task(right).queries
    # Compare stable document targets, never floating ranking scores.
    assert left_target_document_ids == right_target_document_ids


def test_presentation_variants_share_the_same_resolved_retrieval_contract():
    base = ResolvedUserTask(
        factual_query="3731 tropicalisation", exact_entities=("3731",),
        information_need="possibilite de tropicalisation", query_semantics="general_document_question",
        source_constraints=("email", "pdf"), original_factual_query="3731 tropicalisation",
        orchestrator_rewrite="3731 tropicalisation", origin="orchestrator",
    )
    email = ResolvedUserTask(
        factual_query=base.factual_query, exact_entities=base.exact_entities,
        information_need=base.information_need, query_semantics=base.query_semantics,
        response_format="email_draft", target_audience="client",
        source_constraints=base.source_constraints, original_factual_query=base.original_factual_query,
        orchestrator_rewrite=base.orchestrator_rewrite, origin="deterministic_fallback",
    )
    assert base.response_format == "normal"
    assert email.response_format == "email_draft"
    assert_paraphrase_factual_invariance(
        base, email, left_target_document_ids=["email-3730-related"], right_target_document_ids=["email-3730-related"],
    )


def test_fault_injection_contract_keeps_multi_query_retrieval_for_timeout_and_validation_error():
    success = ResolvedUserTask(
        factual_query="Quelle est la bande morte du PS AMS ?", exact_entities=("AMS",), information_need="dead band",
        query_semantics="general_document_question", original_factual_query="Quelle est la bande morte du PS AMS ?",
        orchestrator_rewrite="PS AMS dead band", origin="orchestrator",
    )
    for origin in ("orchestrator_timeout", "orchestrator_validation_error"):
        recovered = ResolvedUserTask(
            factual_query=success.factual_query, exact_entities=success.exact_entities,
            information_need=success.information_need, query_semantics=success.query_semantics,
            response_format="email_draft", target_audience="client",
            original_factual_query=success.original_factual_query,
            orchestrator_rewrite=success.orchestrator_rewrite, origin=origin,
        )
        assert_paraphrase_factual_invariance(
            success, recovered, left_target_document_ids=["ps-ams-sheet"], right_target_document_ids=["ps-ams-sheet"],
        )
        assert len(build_retrieval_queries_for_task(recovered).queries) > 1


def test_generic_paraphrase_cases_keep_factual_retrieval_stable():
    cases = [
        ("3731 tropicalisation", ("3731",), "tropicalisation", "email-3730-related"),
        ("PS AMS dead band", ("AMS",), "dead band", "ps-ams-sheet"),
        ("KG2 DN50 prix de vente", ("DN50", "KG2"), "prix de vente", "gefa-price-xlsx"),
        ("demande statut validation", (), "statut validation", "request-decision-email"),
    ]
    for factual_query, entities, information_need, document_id in cases:
        normal = ResolvedUserTask(
            factual_query=factual_query, exact_entities=entities, information_need=information_need,
            query_semantics="current_state" if "statut" in factual_query else "general_document_question",
            original_factual_query=factual_query, orchestrator_rewrite=factual_query,
        )
        email = ResolvedUserTask(
            factual_query=factual_query, exact_entities=entities, information_need=information_need,
            query_semantics=normal.query_semantics, response_format="email_draft", target_audience="client",
            original_factual_query=factual_query, orchestrator_rewrite=factual_query,
        )
        assert_paraphrase_factual_invariance(
            normal, email, left_target_document_ids=[document_id], right_target_document_ids=[document_id],
        )


def test_fallback_task_adds_canonical_information_need_without_replacing_factual_query():
    task = ResolvedUserTask(
        factual_query="le positionneur 3731 peut être tropicalisé", exact_entities=("3731",),
        information_need="le positionneur 3731 peut être tropicalisé",
        canonical_information_need_query="positionneur 3731 tropicalisation",
        response_format="email_draft", target_audience="client",
        origin="deterministic_fallback",
    )
    queries = build_retrieval_queries_for_task(task).queries
    assert ("original_autonomous", task.factual_query) in queries
    assert ("canonical_information_need", "positionneur 3731 tropicalisation") in queries
