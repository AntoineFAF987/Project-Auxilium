import sys
import types
from pathlib import Path

_BACK_ROOT = Path(__file__).resolve().parents[2]
if str(_BACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACK_ROOT))
if "api" not in sys.modules:
    api_package = types.ModuleType("api")
    api_package.__path__ = [str(_BACK_ROOT / "api")]
    sys.modules["api"] = api_package

from api.iterative_retrieval import run_iterative_evidence_retrieval  # noqa: E402


def _row(document_id, uid, text, *, date="2026-01-01T00:00:00Z", source="email", blocks=None, attachments=None, order=0):
    metadata = {"date": date, "chronological_key": date}
    if attachments is not None:
        metadata["attachments"] = attachments
    return {
        "document_id": document_id, "chunk_uid": uid, "chunk_id": order, "order": order,
        "source": source, "file": f"{document_id}.txt", "path": f"/{document_id}.txt",
        "text": text, "blocks": blocks or [], "document_metadata": metadata,
    }


def test_current_state_prioritizes_latest_state_changing_evidence():
    pending = _row("request", "c1", "Request submitted and pending.", date="2026-01-10T00:00:00Z")
    waiting = _row("request", "c2", "Request is still pending.", date="2026-01-15T00:00:00Z", order=1)
    approved = _row("request", "c3", "Request approved.", date="2026-01-20T00:00:00Z", order=2)
    evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, pending), (0.8, waiting), (0.7, approved)], initial_evidence=[(0.9, pending), (0.8, waiting), (0.7, approved)], corpus=[pending, waiting, approved],
        query="What is the current status?", semantics="current_state",
    )
    assert evidence[0][1]["chunk_uid"] == "c3"
    assert decision.sufficient is True
    assert rounds[0]["action"]["type"] == "SEARCH"


def test_decision_lead_outside_initial_evidence_blocks_sufficiency_and_is_followed():
    submitted = _row("request", "jan10", "Request submitted.", date="2026-01-10T00:00:00Z")
    waiting = _row("request", "jan15", "Request is still waiting.", date="2026-01-15T00:00:00Z", order=1)
    attachment = {"document_id": "decision-pdf", "relation_type": "attachment"}
    header = _row("email-jan20", "jan20-header", "Subject: Final decision regarding request", date="2026-01-20T00:00:00Z", blocks=[{"block_type": "email_header"}], attachments=[attachment])
    decision_pdf = _row("decision-pdf", "decision-body", "Final decision: request approved.", date="2026-01-20T00:00:00Z", source="pdf")
    evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.90, submitted), (0.88, waiting), (0.30, header)],
        initial_evidence=[(0.90, submitted), (0.88, waiting)],
        corpus=[submitted, waiting, header, decision_pdf], query="Was my request finally approved?", semantics="decision",
    )
    assert any(lead["chunk_uid"] == "jan20-header" for lead in rounds[0]["unresolved_leads"])
    assert rounds[0]["sufficiency"]["sufficient"] is False
    assert rounds[1]["action"]["type"] == "FOLLOW"
    assert any(meta["chunk_uid"] == "decision-body" for _, meta in evidence)
    assert decision.sufficient is True


def test_recent_candidate_without_topic_relation_is_not_followed():
    direct = _row("request", "direct", "Final decision: request approved.", date="2026-01-10T00:00:00Z")
    attachment = {"document_id": "unrelated-pdf", "relation_type": "attachment"}
    unrelated = _row("other", "other-header", "Subject: Inventory delivery update", date="2026-01-20T00:00:00Z", blocks=[{"block_type": "email_header"}], attachments=[attachment])
    pdf = _row("unrelated-pdf", "other-pdf", "Inventory delivery details.", source="pdf")
    _evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, direct), (0.3, unrelated)], initial_evidence=[(0.9, direct)],
        corpus=[direct, unrelated, pdf], query="Was my request approved?", semantics="decision",
    )
    assert rounds[0]["unresolved_leads"] == []
    assert len(rounds) == 1
    assert decision.sufficient is True


def test_header_only_email_expands_its_existing_body_before_generation():
    header = _row("email-1", "header", "Subject: Decision regarding Project X", blocks=[{"block_type": "email_header"}])
    body = _row("email-1", "body", "The final decision is approved.", blocks=[{"block_type": "email_body"}], order=1)
    evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, header)], initial_evidence=[(0.9, header)], corpus=[header, body], query="What decision was made?", semantics="decision",
    )
    assert any(meta["chunk_uid"] == "body" for _, meta in evidence)
    assert rounds[1]["action"]["type"] == "EXPAND"
    assert decision.sufficient is True


def test_email_attachment_is_followed_only_when_declared_and_indexed():
    attachment = {"document_id": "decision-pdf", "relation_type": "attachment"}
    header = _row("email-2", "header", "Subject: Final decision attached", blocks=[{"block_type": "email_header"}], attachments=[attachment])
    pdf = _row("decision-pdf", "pdf-1", "Final decision: request approved.", source="pdf")
    evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, header)], initial_evidence=[(0.9, header)], corpus=[header, pdf], query="What was the final decision?", semantics="decision",
    )
    assert any(meta["document_id"] == "decision-pdf" for _, meta in evidence)
    assert rounds[1]["action"]["type"] == "FOLLOW"
    assert rounds[1]["action"]["relation"] == "attachment"
    assert decision.sufficient is True


def test_no_relation_stops_without_repeating_actions():
    header = _row("email-3", "header", "Subject: Project X", blocks=[{"block_type": "email_header"}])
    _evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, header)], initial_evidence=[(0.9, header)], corpus=[header], query="What was decided?", semantics="decision", max_rounds=3,
    )
    assert len(rounds) == 1
    assert decision.sufficient is False
    assert decision.reason == "relevant_document_is_partial_without_new_relation"


def test_same_thread_follow_uses_only_existing_thread_metadata():
    header = _row("email-5", "header", "Subject: Project X", blocks=[{"block_type": "email_header"}])
    header["document_metadata"]["thread_id"] = "thread-1"
    related = _row("email-6", "related", "The request is approved.")
    related["document_metadata"]["thread_id"] = "thread-1"
    evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, header)], initial_evidence=[(0.9, header)], corpus=[header, related], query="What was decided?", semantics="decision",
    )
    assert any(meta["chunk_uid"] == "related" for _, meta in evidence)
    assert rounds[1]["action"]["relation"] == "same_thread"
    assert decision.sufficient is True


def test_round_budget_and_expansion_deduplication_are_bounded():
    header = _row("email-4", "header", "Subject: Project X", blocks=[{"block_type": "email_header"}])
    body = _row("email-4", "body", "Body content.", blocks=[{"block_type": "email_body"}], order=1)
    _evidence, rounds, _decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, header)], initial_evidence=[(0.9, header)], corpus=[header, body], query="Decision?", semantics="decision", max_rounds=3,
    )
    assert len(rounds) <= 3
    assert [round_["action"]["type"] for round_ in rounds].count("EXPAND") <= 1
