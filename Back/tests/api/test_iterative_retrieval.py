import sys
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

from api.iterative_retrieval import complete_same_document_evidence, inspect_attachment_state, run_iterative_evidence_retrieval  # noqa: E402
from rag_core.retrieval import clip_context_blocks, format_context_for_llm  # noqa: E402


def _row(document_id, uid, text, *, date="2026-01-01T00:00:00Z", source="email", blocks=None, attachments=None, order=0, next_chunk_uid=None):
    metadata = {"date": date, "chronological_key": date}
    if attachments is not None:
        metadata["attachments"] = attachments
    return {
        "document_id": document_id, "chunk_uid": uid, "chunk_id": order, "order": order,
        "source": source, "file": f"{document_id}.txt", "path": f"/{document_id}.txt",
        "text": text, "blocks": blocks or [], "document_metadata": metadata, "next_chunk_uid": next_chunk_uid,
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


def test_weak_general_evidence_cannot_hide_direct_subject_candidate():
    unrelated = _row("newsletter", "newsletter-1", "Weekly product newsletter.", source="file")
    matching = _row("orion-pdf", "orion-1", "Project Orion is the programme described here.", source="pdf")
    matching["title"] = "Guide de Project Orion"
    evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, unrelated), (0.5, matching)], initial_evidence=[(0.9, unrelated)],
        corpus=[unrelated, matching], query="What is Project Orion?", semantics="general_document_question",
    )
    assert rounds[0]["sufficiency"]["sufficient"] is False
    assert rounds[0]["candidate_leads"][0]["document_id"] == "orion-pdf"
    assert any(meta["document_id"] == "orion-pdf" for _, meta in evidence)
    assert decision.sufficient is True


def test_weak_general_evidence_without_direct_candidate_is_not_sufficient():
    unrelated = _row("newsletter", "newsletter-1", "Weekly product newsletter.", source="file")
    _evidence, _rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, unrelated)], initial_evidence=[(0.9, unrelated)], corpus=[unrelated],
        query="What is Project Orion?", semantics="general_document_question",
    )
    assert decision.sufficient is False


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
    actions = []
    evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, header)], initial_evidence=[(0.9, header)], corpus=[header, body], query="What decision was made?", semantics="decision", on_action=actions.append,
    )
    assert any(meta["chunk_uid"] == "body" for _, meta in evidence)
    assert rounds[1]["action"]["type"] == "EXPAND"
    assert actions[0].type == "EXPAND"
    assert decision.sufficient is True
    assert decision.critical_structural_evidence_incomplete is False


def test_same_document_completion_adds_the_decisive_neighbour_of_a_topical_body_hit():
    question = _row("tropicalisation", "question", "Can 3730 still be tropicalised?", order=0)
    answer = _row("tropicalisation", "answer", "Tropicalisation is no longer possible.", order=1)
    question["best_topic_relation_score"] = 0.9
    evidence, trace = complete_same_document_evidence([(0.95, question)], [question, answer])
    assert trace["triggered"] is True
    assert trace["added_chunk_uids"] == ["answer"]
    assert [meta["chunk_uid"] for _, meta in evidence] == ["question", "answer"]


def test_same_document_completion_is_bounded_to_immediate_neighbours():
    rows = [_row("long", f"c{index}", f"chunk {index}", order=index) for index in range(7)]
    rows[3]["best_topic_relation_score"] = 0.9
    evidence, trace = complete_same_document_evidence([(1.0, rows[3])], rows, max_neighbors=2)
    assert trace["added_chunk_uids"] == ["c2", "c4"]
    assert len(evidence) == 3


def test_header_without_body_is_a_critical_structural_gap():
    header = _row("email-no-body", "header-only", "Subject: Decision", blocks=[{"block_type": "email_header"}])
    _evidence, _rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, header)], initial_evidence=[(0.9, header)], corpus=[header],
        query="What decision was made in this email?", semantics="decision",
    )
    assert decision.critical_structural_evidence_incomplete is True
    assert decision.critical_structural_gaps[0]["gap_type"] == "email_header_only"


def test_optional_old_email_does_not_block_new_direct_decision():
    old = _row("old", "old-body", "The decision is pending.", date="2026-08-01T00:00:00Z")
    recent = _row("recent", "recent-body", "The request is approved.", date="2026-08-28T00:00:00Z")
    old_extra = _row("old", "old-extra", "Signature", date="2026-08-01T00:00:00Z", order=1)
    _evidence, _rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, recent), (0.7, old)], initial_evidence=[(0.9, recent), (0.7, old)],
        corpus=[recent, old, old_extra], query="What is the final decision?", semantics="decision",
    )
    assert decision.critical_structural_evidence_incomplete is False
    assert decision.optional_structural_evidence_remaining is True


def test_newer_uninspected_decision_lead_is_critical():
    old = _row("request", "old", "Project Atlas decision was pending.", date="2026-08-01T00:00:00Z")
    newer = _row("newer", "newer-header", "Project Atlas", date="2026-08-28T00:00:00Z", blocks=[{"block_type": "email_header"}])
    _evidence, _rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, old), (0.8, newer)], initial_evidence=[(0.9, old)],
        corpus=[old, newer], query="What is the final decision for Project Atlas?",
        semantics="decision", max_rounds=1,
    )
    assert decision.critical_structural_evidence_incomplete is True
    assert any(item["criticality_reason"] == "newer_state_resolution_required" for item in decision.critical_structural_gaps)




def test_august_email_header_expansion_places_the_body_in_final_context():
    """Regression: a header hit must carry its adjacent indexed body to the LLM."""
    body_text = "Madame KOENIG a accueilli favorablement notre requête, en levant votre interdiction de quitter le territoire national à compter de ce jour."
    header = _row(
        "email-2026-08-28", "aug28-header",
        "Subject: Modification de votre contrôle judiciaire et levée de votre interdiction de quitter le territoire national\nFrom: Cabinet",
        date="2026-08-28T00:00:00Z", blocks=[{"block_type": "email_header"}], next_chunk_uid="aug28-body",
    )
    body = _row(
        "email-2026-08-28", "aug28-body", body_text,
        date="2026-08-28T00:00:00Z", blocks=[{"block_type": "email_body"}], order=1,
    )
    evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.95, header)], initial_evidence=[(0.95, header)], corpus=[header, body],
        query="Que dit ce mail concernant la décision ?", semantics="decision",
    )

    final_context = "\n".join(meta["text"] for _, meta in evidence)
    assert rounds[0]["evidence_gaps"][0]["gap_type"] == "email_header_only"
    assert rounds[1]["action"]["type"] == "EXPAND"
    assert rounds[1]["added_chunk_uids"] == ["aug28-body"]
    assert "levant votre interdiction de quitter le territoire national à compter de ce jour" in final_context
    assert decision.sufficient is True


def test_email_attachment_is_followed_only_when_declared_and_indexed():
    attachment = {"document_id": "decision-pdf", "relation_type": "attachment"}
    header = _row("email-2", "header", "Subject: Final decision attached", blocks=[{"block_type": "email_header"}], attachments=[attachment])
    pdf = _row("decision-pdf", "pdf-1", "Final decision: request approved.", source="pdf")
    actions = []
    evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, header)], initial_evidence=[(0.9, header)], corpus=[header, pdf], query="What was the final decision?", semantics="decision", on_action=actions.append,
    )
    assert any(meta["document_id"] == "decision-pdf" for _, meta in evidence)
    assert rounds[1]["action"]["type"] == "FOLLOW"
    assert rounds[1]["action"]["relation"] == "attachment"
    assert actions[0].relation == "attachment"
    assert decision.sufficient is True


def test_attachment_state_distinguishes_reference_from_indexed_content():
    attachment = {"document_id": "missing-pdf", "relation_type": "attachment", "path": "/not-present.pdf"}
    header = _row("email", "header", "Subject: Decision", attachments=[attachment])

    state = inspect_attachment_state(header, [header], attachment_document_id="missing-pdf")

    assert state["attachment_reference_found"] is True
    assert state["attachment_file_found"] is False
    assert state["attachment_parsed"] is False
    assert state["attachment_text_available"] is False
    assert state["attachment_indexed"] is False
    assert state["attachment_chunk_count"] == 0


def test_explicit_attachment_request_follows_direct_document_id():
    attachment = {"document_id": "decision-pdf", "relation_type": "attachment"}
    header = _row("email", "header", "Subject: Decision", attachments=[attachment])
    body = _row("email", "body", "Please see attached decision.", blocks=[{"block_type": "email_body"}], order=1)
    pdf = _row("decision-pdf", "pdf", "Decision content.", source="pdf")

    evidence, rounds, _decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, header)], initial_evidence=[(0.9, header)],
        corpus=[header, body, pdf], query="Que dit l'annexe jointe ?", semantics="general_document_question",
    )

    assert any(meta["document_id"] == "decision-pdf" for _, meta in evidence)
    assert any(round_["action"].get("relation") == "attachment" for round_ in rounds)


def test_explicit_attachment_request_reports_unavailable_content_without_search():
    attachment = {"document_id": "missing-pdf", "relation_type": "attachment"}
    header = _row("email", "header", "Subject: Decision", attachments=[attachment])
    body = _row("email", "body", "Please see attached decision.", blocks=[{"block_type": "email_body"}], order=1)

    _evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, header)], initial_evidence=[(0.9, header)],
        corpus=[header, body], query="Que dit l'annexe jointe ?", semantics="general_document_question",
    )

    assert decision.sufficient is False
    assert decision.reason == "attachment_content_unavailable"
    assert rounds[-1]["attachment_states"][0]["attachment_indexed"] is False


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


def test_selected_header_is_an_evidence_gap_and_expands_before_unrelated_candidate():
    pending = _row("request", "jan10", "Request remains pending.", date="2026-01-10T00:00:00Z")
    attachment = {"document_id": "final-pdf", "relation_type": "attachment"}
    header = _row("email-jan20", "jan20-header", "Subject: Final decision regarding request", date="2026-01-20T00:00:00Z", blocks=[{"block_type": "email_header"}], attachments=[attachment])
    body = _row("email-jan20", "jan20-body", "The request was approved.", date="2026-01-20T00:00:00Z", blocks=[{"block_type": "email_body"}], order=1)
    unrelated_attachment = {"document_id": "inventory-pdf", "relation_type": "attachment"}
    unrelated = _row("inventory", "other-header", "Subject: Inventory delivery update", date="2026-01-21T00:00:00Z", blocks=[{"block_type": "email_header"}], attachments=[unrelated_attachment])
    pdf = _row("final-pdf", "pdf", "Final attached decision.", source="pdf")
    unrelated_pdf = _row("inventory-pdf", "other-pdf", "Inventory details.", source="pdf")

    evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.90, pending), (0.35, header), (0.34, unrelated)],
        initial_evidence=[(0.90, pending), (0.35, header)],
        corpus=[pending, header, body, unrelated, pdf, unrelated_pdf],
        query="Was my request finally approved?", semantics="decision",
    )

    assert any(gap["document_id"] == "email-jan20" for gap in rounds[0]["evidence_gaps"])
    assert rounds[0]["selected_lead_type"] == "evidence_gap"
    assert rounds[1]["action"]["type"] == "EXPAND"
    assert rounds[1]["selected_lead"] == "email-jan20"
    assert not any(item[1]["chunk_uid"] == "other-pdf" for item in evidence)
    assert decision.sufficient is True


def test_selected_header_expands_then_follows_attachment_when_body_is_not_outcome_evidence():
    attachment = {"document_id": "final-pdf", "relation_type": "attachment"}
    header = _row("email", "header", "Subject: Final decision regarding request", blocks=[{"block_type": "email_header"}], attachments=[attachment])
    body = _row("email", "body", "Please see attached final decision.", blocks=[{"block_type": "email_body"}], order=1)
    pdf = _row("final-pdf", "pdf", "The request was approved.", source="pdf")
    evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, header)], initial_evidence=[(0.9, header)], corpus=[header, body, pdf],
        query="Was my request finally approved?", semantics="decision",
    )

    assert [round_["action"]["type"] for round_ in rounds] == ["SEARCH", "EXPAND", "FOLLOW"]
    assert rounds[2]["action"]["relation"] == "attachment"
    assert any(meta["chunk_uid"] == "pdf" for _, meta in evidence)
    assert decision.sufficient is True


def test_loaded_header_without_available_relation_is_not_actionable_gap():
    header = _row("email", "header", "Subject: Project update", blocks=[{"block_type": "email_header"}])
    _evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, header)], initial_evidence=[(0.9, header)], corpus=[header],
        query="What was decided?", semantics="decision",
    )
    assert rounds[0]["evidence_gaps"] == []
    assert len(rounds) == 1
    assert decision.sufficient is False


def test_current_state_prioritizes_newer_state_resolution_gap_over_higher_retrieval_scores():
    jan10 = _row("jan10", "jan10-header", "Subject: Request submitted", date="2026-01-10T00:00:00Z", blocks=[{"block_type": "email_header"}])
    jan10_body = _row("jan10", "jan10-body", "The request was submitted.", date="2026-01-10T00:00:00Z", blocks=[{"block_type": "email_body"}], order=1)
    jan15 = _row("jan15", "jan15-header", "Subject: Request pending", date="2026-01-15T00:00:00Z", blocks=[{"block_type": "email_header"}])
    jan15_body = _row("jan15", "jan15-body", "The request is pending.", date="2026-01-15T00:00:00Z", blocks=[{"block_type": "email_body"}], order=1)
    jan20 = _row("jan20", "jan20-header", "Subject: Status update for request", date="2026-01-20T00:00:00Z", blocks=[{"block_type": "email_header"}])
    jan20_body = _row("jan20", "jan20-body", "The request was approved.", date="2026-01-20T00:00:00Z", blocks=[{"block_type": "email_body"}], order=1)

    _evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.95, jan10), (0.90, jan15), (0.30, jan20)],
        initial_evidence=[(0.95, jan10), (0.90, jan15), (0.30, jan20)],
        corpus=[jan10, jan10_body, jan15, jan15_body, jan20, jan20_body],
        query="What is the current status of my request?", semantics="current_state",
    )

    assert rounds[0]["selected_lead"] == "jan20"
    selected_gap = next(gap for gap in rounds[0]["evidence_gaps"] if gap["document_id"] == "jan20")
    assert selected_gap["priority_class"] == "potential_newer_state_resolution"
    assert selected_gap["state_change_potential"] == "unknown"
    assert rounds[1]["action"]["target_document_id"] == "jan20"
    assert decision.sufficient is True


def test_current_state_does_not_promote_newer_state_header_from_another_topic():
    request = _row("request", "request-header", "Subject: Request submitted", date="2026-01-10T00:00:00Z", blocks=[{"block_type": "email_header"}])
    request_body = _row("request", "request-body", "Request remains pending.", date="2026-01-10T00:00:00Z", blocks=[{"block_type": "email_body"}], order=1)
    unrelated = _row("inventory", "inventory-header", "Subject: Final status update for inventory", date="2026-01-20T00:00:00Z", blocks=[{"block_type": "email_header"}])
    unrelated_body = _row("inventory", "inventory-body", "Inventory was approved.", date="2026-01-20T00:00:00Z", blocks=[{"block_type": "email_body"}], order=1)

    _evidence, rounds, _decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.95, request), (0.30, unrelated)], initial_evidence=[(0.95, request), (0.30, unrelated)],
        corpus=[request, request_body, unrelated, unrelated_body],
        query="What is the current status of my request?", semantics="current_state",
    )

    unrelated_gap = next(gap for gap in rounds[0]["evidence_gaps"] if gap["document_id"] == "inventory")
    assert unrelated_gap["semantic_fit"] == "weak_selected_evidence"
    assert unrelated_gap["priority_class"] != "potential_newer_state_resolution"
    assert rounds[0]["selected_lead"] != "inventory"


def test_dominant_document_is_frozen_as_anchor_and_survives_round_budget():
    anchor_a = _row("guide", "guide-1", "Project Atlas supports the requested capability.", source="pdf")
    anchor_b = _row("guide", "guide-2", "The capability is described in the technical guide.", source="pdf", order=1)
    weak = _row("weak", "weak-1", "Unrelated generic material.", source="pdf")
    evidence, rounds, _decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.95, anchor_a), (0.91, anchor_b), (0.08, weak)],
        initial_evidence=[(0.95, anchor_a), (0.91, anchor_b)], corpus=[anchor_a, anchor_b, weak],
        query="Does Project Atlas support the requested capability?", semantics="fact_lookup", max_rounds=1,
    )
    assert any(item["document_id"] == "guide" for item in rounds[0]["anchor_documents"])
    assert {"guide-1", "guide-2"}.issubset({meta["chunk_uid"] for _, meta in evidence})
    assert "guide-1" in rounds[-1]["core_evidence_chunk_uids"]


def test_answerable_anchor_stops_with_optional_relation_available():
    anchor = _row("guide", "guide-1", "Project Atlas supports the requested capability.", source="pdf")
    adjacent = _row("guide", "guide-2", "Supplementary implementation detail.", source="pdf", order=1)
    _evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.95, anchor)], initial_evidence=[(0.95, anchor)], corpus=[anchor, adjacent],
        query="Does Project Atlas support the requested capability?", semantics="fact_lookup",
    )
    assert decision.sufficient is True
    assert len(rounds) == 1
    assert "optional_expansions_remaining" in decision.reason


def test_weak_expandable_candidate_is_not_a_lead_when_anchor_answers_question():
    anchor = _row("guide", "guide-1", "Project Atlas supports the requested capability.", source="pdf")
    weak = _row("weak", "weak-1", "Atlas appears once in unrelated material.", source="pdf")
    weak_adjacent = _row("weak", "weak-2", "Additional unrelated material.", source="pdf", order=1)
    _evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(1.0, anchor), (0.05, weak)], initial_evidence=[(1.0, anchor)], corpus=[anchor, weak, weak_adjacent],
        query="Does Project Atlas support the requested capability?", semantics="fact_lookup",
    )
    assert decision.sufficient is True
    assert rounds[0]["candidate_leads"] == []


def test_newer_related_state_body_is_promoted_after_expand():
    initial = _row("initial", "initial-1", "Request submitted.", date="2026-01-01T00:00:00Z")
    waiting = _row("waiting", "waiting-1", "Request remains pending.", date="2026-01-05T00:00:00Z")
    header = _row("latest", "latest-header", "Subject: Final status update for request", date="2026-01-10T00:00:00Z", blocks=[{"block_type": "email_header"}])
    body = _row("latest", "latest-body", "The request is approved.", date="2026-01-10T00:00:00Z", blocks=[{"block_type": "email_body"}], order=1)
    evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.95, waiting), (0.90, initial), (0.35, header)],
        initial_evidence=[(0.95, waiting), (0.90, initial), (0.35, header)], corpus=[initial, waiting, header, body],
        query="What is the current status of my request?", semantics="current_state",
    )
    assert "latest-body" in rounds[-1]["core_evidence_chunk_uids"]
    assert any("latest-body" in round_.get("state_change_promoted_chunks", []) for round_ in rounds)
    assert any(meta["chunk_uid"] == "latest-body" for _, meta in evidence)
    assert decision.sufficient is True


def test_expanded_latest_body_is_reevaluated_promoted_and_precedes_its_header():
    """A generic temporal regression: the only latest content arrives via EXPAND."""
    old = _row("record-a", "old", "Service request status: pending.", date="2026-01-01T00:00:00Z")
    middle = _row("record-b", "middle", "Service request: no determination yet.", date="2026-01-10T00:00:00Z")
    header = _row(
        "record-c", "latest-header", "Subject: Service request correspondence",
        date="2026-01-20T00:00:00Z", blocks=[{"block_type": "email_header"}], next_chunk_uid="latest-body",
    )
    body = _row(
        "record-c", "latest-body", "Service request status is now approved.",
        date="2026-01-20T00:00:00Z", blocks=[{"block_type": "email_body"}], order=1,
    )
    evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.95, old), (0.90, middle), (0.30, header)],
        initial_evidence=[(0.95, old), (0.90, middle), (0.30, header)],
        corpus=[old, middle, header, body], query="What is the current status of the service request?", semantics="current_state",
    )

    expanded_round = next(round_ for round_ in rounds if round_["action"]["type"] == "EXPAND")
    assert "latest-body" in expanded_round["state_change_promoted_chunks"]
    assert expanded_round["expanded_chunk_reevaluation"][0]["state_change_potential_before_expand"] == "unknown"
    assert expanded_round["expanded_chunk_reevaluation"][0]["state_change_potential_after_expand"] == "true"
    assert evidence[0][1]["chunk_uid"] == "latest-body"
    assert decision.sufficient is True


def test_expanded_latest_body_without_a_new_outcome_is_still_evaluated_as_current_evidence():
    old = _row("record-a", "old", "Service request status: pending.", date="2026-01-01T00:00:00Z")
    header = _row(
        "record-c", "latest-header", "Subject: Service request correspondence",
        date="2026-01-20T00:00:00Z", blocks=[{"block_type": "email_header"}], next_chunk_uid="latest-body",
    )
    body = _row(
        "record-c", "latest-body", "Service request remains pending.",
        date="2026-01-20T00:00:00Z", blocks=[{"block_type": "email_body"}], order=1,
    )
    evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.95, old), (0.30, header)], initial_evidence=[(0.95, old), (0.30, header)],
        corpus=[old, header, body], query="What is the current status of the service request?", semantics="current_state",
    )

    assert evidence[0][1]["chunk_uid"] == "latest-body"
    assert any("latest-body" in round_.get("state_change_promoted_chunks", []) for round_ in rounds)
    assert decision.sufficient is True


def test_candidate_shared_person_name_is_not_a_topical_relation():
    direct = _row("request", "direct", "Request approved.", date="2026-01-10T00:00:00Z")
    attachment = {"document_id": "application-pdf", "relation_type": "attachment"}
    other_personal = _row("application", "application-header", "Subject: Application at Example Corp for Alex Doe", date="2026-01-20T00:00:00Z", blocks=[{"block_type": "email_header"}], attachments=[attachment])
    pdf = _row("application-pdf", "application-pdf-chunk", "Application details.", source="pdf")

    _evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.9, direct), (0.4, other_personal)], initial_evidence=[(0.9, direct)],
        corpus=[direct, other_personal, pdf], query="What is the current status of Alex Doe's request?", semantics="current_state",
    )

    assert rounds[0]["candidate_leads"] == []
    assert decision.sufficient is True


def test_email_body_expansion_promotes_only_the_decision_chunk_from_a_shared_native_block():
    """A shared parent block must not leak its decision into a signature."""
    native_body = (
        "Madame KOENIG a accueilli favorablement la requête et a levé l'interdiction de quitter le territoire.\n\n"
        "Les autres obligations restent inchangées.\n\n"
        "Je me tiens à votre disposition. Bien à vous."
    )
    header = _row(
        "recent-mail", "recent-header", "Subject: Modification et levée de l'interdiction de quitter le territoire",
        date="2026-08-28T12:46:34Z", blocks=[{"block_type": "email_header"}],
        next_chunk_uid="recent-decision",
    )
    decision_body = _row(
        "recent-mail", "recent-decision",
        "Madame KOENIG a accueilli favorablement la requête et a levé l'interdiction de quitter le territoire.",
        date="2026-08-28T12:46:34Z", blocks=[{"block_type": "email_body", "text": native_body}], order=1,
        next_chunk_uid="recent-obligations",
    )
    obligations = _row(
        "recent-mail", "recent-obligations", "Les autres obligations restent inchangées.",
        date="2026-08-28T12:46:34Z", blocks=[{"block_type": "email_body", "text": native_body}], order=2,
        next_chunk_uid="recent-signature",
    )
    signature = _row(
        "recent-mail", "recent-signature",
        "Je me tiens à votre disposition. Bien à vous.",
        date="2026-08-28T12:46:34Z", blocks=[{"block_type": "email_body", "text": native_body}], order=3,
    )
    evidence, rounds, decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.95, header), (0.93, decision_body)], initial_evidence=[(0.95, header)],
        corpus=[header, decision_body, obligations, signature],
        query="Antoine demande par mail s'il est autorisé à quitter le territoire. Je réponds quoi ?",
        semantics="decision",
    )

    expanded_round = next(round_ for round_ in rounds if round_["action"]["type"] == "EXPAND")
    by_uid = {item["chunk_uid"]: item for item in expanded_round["expanded_chunk_reevaluation"]}
    final_context = format_context_for_llm(clip_context_blocks(evidence, keep=10))
    assert rounds[1]["action"]["type"] == "EXPAND"
    assert decision.sufficient is True
    assert expanded_round["state_change_promoted_chunks"] == ["recent-decision"]
    assert {item["evaluated_text_source"] for item in by_uid.values()} == {"chunk.text"}
    assert by_uid["recent-decision"]["state_change_potential_after_expand"] == "true"
    assert by_uid["recent-obligations"]["state_change_potential_after_expand"] == "false"
    assert by_uid["recent-signature"]["state_change_potential_after_expand"] == "false"
    assert "a levé l'interdiction de quitter le territoire" in final_context


def test_expand_reevaluates_decision_already_represented_by_fused_evidence():
    header = _row(
        "mail", "header", "Subject: Travel restriction decision",
        blocks=[{"block_type": "email_header"}], next_chunk_uid="decision",
    )
    decision = _row(
        "mail", "decision", "The travel restriction is lifted effective today.",
        blocks=[{"block_type": "email_body"}], order=1, next_chunk_uid="signature",
    )
    signature = _row(
        "mail", "signature", "Kind regards.", blocks=[{"block_type": "email_body"}], order=2,
    )
    fused_header_and_decision = {
        **header,
        "text": f"{header['text']}\n\n{decision['text']}",
        "fused_chunk_uids": ["header", "decision"],
    }
    evidence, rounds, _decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.95, header), (0.93, decision)],
        initial_evidence=[(0.94, fused_header_and_decision)],
        corpus=[header, decision, signature],
        query="Is the travel restriction lifted?", semantics="decision",
    )

    expanded_round = next(round_ for round_ in rounds if round_["action"]["type"] == "EXPAND")
    reviewed = {item["chunk_uid"]: item for item in expanded_round["expanded_chunk_reevaluation"]}
    final_context = format_context_for_llm(clip_context_blocks(evidence, keep=10))
    assert expanded_round["expanded_document_chunks_considered"] == ["decision", "signature"]
    assert expanded_round["expanded_document_chunks_promoted"] == ["decision"]
    assert reviewed["decision"]["already_represented_in_evidence"] is True
    assert "The travel restriction is lifted effective today." in final_context


def test_expansion_never_promotes_signature_from_parent_block_text():
    header = _row(
        "mail", "header", "Subject: Decision about lifting the restriction", blocks=[{"block_type": "email_header"}],
        next_chunk_uid="signature",
    )
    signature = _row(
        "mail", "signature", "Bien à vous.",
        blocks=[{"block_type": "email_body", "text": "Decision approved and restriction lifted. Bien à vous."}], order=1,
    )
    _evidence, rounds, _decision = run_iterative_evidence_retrieval(
        candidate_pool=[(0.95, header)], initial_evidence=[(0.95, header)],
        corpus=[header, signature],
        query="Quelle décision ce mail communique-t-il ?", semantics="decision",
    )

    expanded_round = next(round_ for round_ in rounds if round_["action"]["type"] == "EXPAND")
    rechecked_signature = expanded_round["expanded_chunk_reevaluation"][0]
    assert expanded_round["state_change_promoted_chunks"] == []
    assert rechecked_signature["chunk_uid"] == "signature"
    assert rechecked_signature["state_change_potential_after_expand"] == "false"
