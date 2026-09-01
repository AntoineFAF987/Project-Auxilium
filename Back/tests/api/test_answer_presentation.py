import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).parents[2] / "api" / "answer_presentation.py"
_SPEC = importlib.util.spec_from_file_location("answer_presentation", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
extract_citations_and_clean_answer = _MODULE.extract_citations_and_clean_answer
remap_inline_citations = _MODULE.remap_inline_citations
select_web_source_indices = _MODULE.select_web_source_indices
enforce_final_citation_contract = _MODULE.enforce_final_citation_contract


def test_extracts_private_citations_and_removes_display_markers():
    answer, citations = extract_citations_and_clean_answer(
        "Los Angeles est le meilleur choix. [1][2]\n<CITATIONS>[2,1,2]</CITATIONS>"
    )
    assert answer == "Los Angeles est le meilleur choix. [1][2]"
    assert citations == [2, 1]


def test_preserves_code_identifiers_and_legitimate_brackets():
    answer, citations = extract_citations_and_clean_answer(
        "Utilise `model_v2[1]` et la formule array[1].\n<CITATIONS>[1]</CITATIONS>"
    )
    assert answer == "Utilise `model_v2[1]` et la formule array[1]."
    assert citations == [1]


@pytest.mark.parametrize(
    ("raw", "expected", "expected_citations"),
    [
        ("Réponse courte sans source.", "Réponse courte sans source.", []),
        ("Synthèse locale. [1][2]\n<CITATIONS>[1,2]</CITATIONS>", "Synthèse locale. [1][2]", [1, 2]),
        ("Synthèse Web. [1][2][3]\n<CITATIONS>[1,2,3]</CITATIONS>", "Synthèse Web. [1][2][3]", [1, 2, 3]),
        ("- Première étape\n- Seconde étape", "- Première étape\n- Seconde étape", []),
        ("Le produit AX-17_v2 est compatible.", "Le produit AX-17_v2 est compatible.", []),
        ("`model_v2[1]`\n```py\nvalue = array[1]\n```", "`model_v2[1]`\n```py\nvalue = array[1]\n```", []),
        ("**Important** et _léger_.", "**Important** et _léger_.", []),
        ("Premier paragraphe.\n\nSecond paragraphe.", "Premier paragraphe.\n\nSecond paragraphe.", []),
    ],
)
def test_display_cleanup_preserves_legitimate_answer_content(raw, expected, expected_citations):
    answer, citations = extract_citations_and_clean_answer(raw)
    assert answer == expected
    assert citations == expected_citations


def test_inline_chunk_citations_are_remapped_to_final_document_numbers():
    answer = remap_inline_citations("Choisis Los Angeles. [1][3][6][8]", {1: 1, 3: 1, 6: 1, 8: 2})
    assert answer == "Choisis Los Angeles. [1, 2]"
    assert "[1][2]" not in answer
    assert "[3]" not in answer and "[6]" not in answer and "[8]" not in answer


def test_mid_sentence_citation_is_removed_not_left_in_the_reader_text():
    answer = remap_inline_citations("Le 3730 [1] possède déjà un vernis.", {1: 1})
    assert answer == "Le 3730 possède déjà un vernis."


def test_end_of_sentence_citation_is_kept_and_grouped():
    answer = remap_inline_citations("Le 3730 possède déjà un vernis. [1][3]", {1: 1, 3: 2})
    assert answer == "Le 3730 possède déjà un vernis. [1, 2]"


def test_grouped_bracket_citation_and_adjacent_text_are_normalized():
    answer = remap_inline_citations(
        "Information A. [1]Information B. [4, 10]",
        {1: 1, 4: 1, 10: 2},
    )
    assert answer == "Information A. [1]\n\nInformation B. [1, 2]"


def test_already_grouped_raw_chunk_citation_is_remapped_once():
    answer = remap_inline_citations("Les éléments convergent. [3, 6, 8]", {3: 1, 6: 1, 8: 2})
    assert answer == "Les éléments convergent. [1, 2]"


def test_final_contract_removes_out_of_range_citations_from_legacy_or_cache_paths():
    answer = enforce_final_citation_contract(
        "Information A. [1]Information B. [4, 10]", source_count=2,
    )
    assert answer == "Information A. [1]\n\nInformation B."
    assert "[4" not in answer and "10]" not in answer


def test_orphan_citation_is_removed_instead_of_becoming_a_false_reference():
    answer = remap_inline_citations("Cette précision a été écartée. [4]", {1: 1, 2: 2})
    assert answer == "Cette précision a été écartée."


def test_web_citations_use_the_same_final_numbering_contract():
    selected, mapping = select_web_source_indices(8, [1, 3, 6, 8])
    assert selected == [1, 3, 6, 8]
    assert remap_inline_citations("Résultat. [1][3][6][8]", mapping) == "Résultat. [1, 2, 3, 4]"


def test_answer_without_sources_has_no_numeric_citation_marker():
    assert remap_inline_citations("Réponse directe sans source. [1]", {}) == "Réponse directe sans source."


def test_generation_prompts_default_to_direct_non_audit_style():
    for module_name in ("llm.py", "llm_stream.py"):
        prompt_source = (_MODULE_PATH.parents[1] / "rag_core" / module_name).read_text(encoding="utf-8")
        assert "Réponds d'abord à la question" in prompt_source
        assert "sans formule d'audit" in prompt_source
        assert "sans répéter la même réserve" in prompt_source
        assert "sans confondre concision et réponse incomplète" in prompt_source
        assert "question technique, comparative ou de recommandation" in prompt_source
        assert "preuve voisine mais non exacte" in prompt_source
        assert "couvre les éléments distincts et utiles du CONTEXTE" in prompt_source
        assert "exploite chaque critère réellement documenté" in prompt_source
