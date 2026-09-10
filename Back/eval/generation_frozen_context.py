"""Frozen-context generation comparison for prompt-only experiments.

This module deliberately bypasses retrieval.  It captures the production
prompt builder's payload, changes only its system instructions in memory, and
can optionally submit that one final-generation payload to the configured
provider.  It is an evaluation harness, not a production dependency.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from rag_core import llm
from rag_core.llm_providers import generate_from_payload


@dataclass(frozen=True)
class FrozenFixture:
    identifier: str
    question: str
    context: str
    evidence_mode: str
    answerability: str
    exact_entity_guard: bool = False
    missing_exact_entities: tuple[str, ...] = ()
    required_patterns: tuple[tuple[str, ...], ...] = ()
    forbidden_patterns: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class PromptVariant:
    identifier: str
    instruction: str = ""


BASELINE = PromptVariant("baseline")
VARIANTS = (
    BASELINE,
    PromptVariant(
        "A_explicit_completeness",
        " Avant de répondre, identifie silencieusement tous les éléments de preuve qui aident matériellement à répondre à la question. "
        "Donne d'abord la réponse directe. Puis inclus tous les faits matériellement utiles qui expliquent pourquoi, comment, "
        "dans quelles conditions, avec quelles valeurs, limites ou exceptions. Ne t'arrête pas après la première phrase qui répond. "
        "N'inclus aucun détail qui n'aide pas matériellement à répondre.",
    ),
    PromptVariant(
        "B_silent_completeness_check",
        " Avant de finaliser, vérifie silencieusement que tu n'as omis aucune raison, condition, limite, valeur, exception ou mécanisme "
        "explicitement présent dans les preuves et utile pour répondre. N'ajoute aucun fait absent des preuves.",
    ),
    PromptVariant(
        "C_direct_complete",
        " Réponds directement d'abord, puis explique la réponse aussi complètement que les preuves pertinentes le permettent. "
        "Utilise toutes les preuves matériellement pertinentes, sans ajouter de détail hors sujet.",
    ),
)


FIXTURES = (
    FrozenFixture(
        "3730_direct",
        "Quelle est la possibilité de tropicalisation du positionneur 3730 ?",
        "[1] Source: mail 3730\nLa carte électronique du TROVIS 3730 possède déjà un vernis. Ce n'est pas une tropicalisation complète. "
        "Elle permet de tenir les spécifications : –20 à +80 °C pour toutes les exécutions, et –45 à +80 °C avec passage de câble métallique. "
        "La tropicalisation n'est plus possible : le changement de technologie du détecteur de position ne le permet plus. "
        "L'application d'un vernis supplémentaire change son comportement.",
        "direct", "answerable",
        required_patterns=(("tropicalisation", "plus possible"), ("déjà", "vernis"), ("pas", "tropicalisation complète"),
                           ("changement", "technologie", "détecteur"), ("vernis supplémentaire", "comportement")),
        forbidden_patterns=(("tropicalisation", "possible"),),
    ),
    FrozenFixture(
        "3731_related",
        "Le positionneur 3731 peut-il être tropicalisé ?",
        "[1] Source: mail 3730\nPour le TROVIS 3730, la tropicalisation n'est plus possible après un changement de technologie du détecteur de position.",
        "related", "partial", True, ("3731",),
        required_patterns=(("3731", "aucune", "preuve"), ("3730",)),
        forbidden_patterns=(("3731", "n'est plus possible"), ("3731", "ne peut pas")),
    ),
    FrozenFixture(
        "ps_ams_direct",
        "Quelle est la dead band du PS AMS ?",
        "[1] Source: confirmation PS-AMS\nLa dead band des actionneurs PS-AMS est réglable entre 0,5 % et 5 %. "
        "Pour ce réglage, le logiciel PSCS et le kit de paramétrage sont nécessaires.",
        "direct", "answerable",
        required_patterns=(("0,5", "5"), ("pscs",), ("kit", "paramétrage")),
        forbidden_patterns=(("1 %",),),
    ),
    FrozenFixture(
        "current_state_direct",
        "Est-ce que ma demande a été validée ?",
        "[1] Source: email du 20 août\nVotre demande est en attente.\n\n"
        "[2] Source: email décisif du 28 août\nMadame KOENIG a accueilli favorablement votre requête en levant votre interdiction de quitter le territoire national à compter de ce jour. "
        "Vos autres obligations et interdictions demeurent inchangées.",
        "direct", "answerable",
        required_patterns=(("valid",), ("lev", "interdiction"), ("autres", "obligations", "inchang")),
        forbidden_patterns=(("encore en attente",),),
    ),
    FrozenFixture(
        "kg2_sales_price",
        "Quel est le prix de vente du KG2 DN50 ?",
        "[1] Source: prix structurés KG2 DN50\nRéférence : KG22366E050. Prix d'achat HT : 54,21 €. Prix de vente France HT : 197,87 €.",
        "direct", "answerable",
        required_patterns=(("197", "87"), ("vente", "france")),
        forbidden_patterns=(("54,21",),),
    ),
)

# This separate collection is the only corpus eligible for ``--live`` prompt
# experiments.  Keep ``FIXTURES`` above unchanged: it remains available for
# controlled local replay and validation, but must not leave the workspace.
SYNTHETIC_FIXTURES = (
    FrozenFixture(
        "x370_direct_synthetic",
        "Quelle est la possibilité de tropicalisation du positionneur X370 ?",
        "[1] Source: note technique fictive X370\nLa carte électronique du X370 possède déjà un vernis. Ce n'est pas une tropicalisation complète. "
        "Elle satisfait les spécifications : –20 à +80 °C pour toutes les exécutions, et –45 à +80 °C avec presse-étoupe métallique. "
        "La tropicalisation supplémentaire n'est plus possible : l'évolution de la technologie du détecteur de position ne le permet plus. "
        "L'application d'un vernis supplémentaire modifie le comportement du dispositif.",
        "direct", "answerable",
        required_patterns=(("tropicalisation", "plus possible"), ("déjà", "vernis"), ("pas", "tropicalisation complète"),
                           ("évolution", "technologie", "détecteur"), ("vernis supplémentaire", "comportement")),
        forbidden_patterns=(("tropicalisation", "possible"),),
    ),
    FrozenFixture(
        "x371_related_synthetic",
        "Le positionneur X371 peut-il être tropicalisé ?",
        "[1] Source: note technique fictive X370\nPour le X370, la tropicalisation n'est plus possible après une évolution de la technologie du détecteur de position.",
        "related", "partial", True, ("X371",),
        required_patterns=(("x371", "aucune", "preuve"), ("x370",)),
        forbidden_patterns=(("x371", "n'est plus possible"), ("x371", "ne peut pas")),
    ),
    FrozenFixture(
        "ax_ams_direct_synthetic",
        "Quelle est la dead band de l'actionneur AX-AMS ?",
        "[1] Source: confirmation fictive AX-AMS\nLa dead band des actionneurs AX-AMS est réglable entre 0,5 % et 5 %. "
        "Pour ce réglage, le logiciel AXCS et le kit de paramétrage sont nécessaires.",
        "direct", "answerable",
        required_patterns=(("0,5", "5"), ("axcs",), ("kit", "paramétrage")),
        forbidden_patterns=(("1 %",),),
    ),
    FrozenFixture(
        "administrative_current_state_synthetic",
        "Est-ce que ma demande administrative a été validée ?",
        "[1] Source: message antérieur fictif du 3 mars\nVotre demande administrative est en attente.\n\n"
        "[2] Source: décision fictive du 12 mars\nL'autorité a accueilli favorablement votre demande et a levé la restriction administrative à compter de ce jour. "
        "Vos autres obligations administratives demeurent inchangées.",
        "direct", "answerable",
        required_patterns=(("valid",), ("lev", "restriction"), ("autres", "obligations", "inchang")),
        forbidden_patterns=(("encore en attente",),),
    ),
    FrozenFixture(
        "zx2_sales_price_synthetic",
        "Quel est le prix de vente du ZX2 DN50 ?",
        "[1] Source: grille tarifaire fictive ZX2 DN50\nRéférence : ZX22366F050. Prix d'achat HT : 54,21 €. Prix de vente France HT : 197,87 €.",
        "direct", "answerable",
        required_patterns=(("197", "87"), ("vente", "france")),
        forbidden_patterns=(("54,21",),),
    ),
)


def capture_current_payload(fixture: FrozenFixture) -> dict[str, Any]:
    """Build the exact current production payload without calling a provider."""
    captured: dict[str, Any] = {}
    original = llm.generate_from_payload
    try:
        llm.generate_from_payload = lambda payload: captured.update(payload) or "__captured__"
        llm.ask_mistral_with_context(
            fixture.question, fixture.context, history=[], evidence_mode=fixture.evidence_mode,
            answerability=fixture.answerability, exact_entity_guard=fixture.exact_entity_guard,
            missing_exact_entities=list(fixture.missing_exact_entities),
        )
    finally:
        llm.generate_from_payload = original
    return captured


def payload_for_variant(fixture: FrozenFixture, variant: PromptVariant) -> dict[str, Any]:
    payload = capture_current_payload(fixture)
    payload["messages"] = [dict(message) for message in payload["messages"]]
    if variant.instruction:
        payload["messages"][0]["content"] += variant.instruction
    return payload


def _matches(answer: str, terms: Iterable[str]) -> bool:
    text = answer.casefold()
    normalized_terms = tuple(term.casefold() for term in terms)
    if normalized_terms == ("tropicalisation", "plus possible"):
        return bool(re.search(r"tropicalisation.{0,48}(?:n.{0,8}(?:plus |pas )possible)", text))
    if normalized_terms == ("pas", "tropicalisation complète"):
        return bool(re.search(r"(?:pas|sans.{0,24}être|ne bénéficie pas d.{0,8}).{0,48}tropicalisation complète", text))
    if normalized_terms == ("vernis supplémentaire", "comportement"):
        return bool(re.search(r"(?:vernis (?:supplémentaire|additionnel)|ajout d.{0,16}vernis).{0,80}(?:modifi|chang).{0,40}comportement", text))
    if normalized_terms == ("tropicalisation", "possible"):
        return bool(re.search(r"tropicalisation.{0,48}(?<!plus )(?<!pas )possible", text))
    if normalized_terms in {("x371", "aucune", "preuve"), ("3731", "aucune", "preuve")}:
        entity = normalized_terms[0]
        return bool(re.search(
            rf"(?:aucun(?:e)? (?:preuve|information|élément) direct.{{0,64}}{entity}|"
            rf"{entity}.{{0,64}}aucun(?:e)? (?:preuve|information|élément) direct|"
            rf"aucun élément ne confirme.{{0,80}}{entity}|"
            rf"{entity}.{{0,64}}sans preuve directe)", text
        ))
    if normalized_terms in {("x371", "n'est plus possible"), ("3731", "n'est plus possible")}:
        return bool(re.search(rf"{normalized_terms[0]}.{{0,40}}n'est plus possible", text))
    if normalized_terms in {("x371", "ne peut pas"), ("3731", "ne peut pas")}:
        return bool(re.search(rf"{normalized_terms[0]}.{{0,40}}ne peut pas (?:être )?tropicalis", text))
    if normalized_terms == ("valid",):
        return bool(re.search(r"\b(?:valid|oui\b|accept.{0,24}favorabl|accueill.{0,24}favorabl)", text))
    return all(re.search(re.escape(term), text) for term in normalized_terms)


def evaluate_answer(fixture: FrozenFixture, answer: str) -> dict[str, Any]:
    required = [_matches(answer, alternatives) for alternatives in fixture.required_patterns]
    forbidden = [_matches(answer, alternatives) for alternatives in fixture.forbidden_patterns]
    return {
        "required_fact_recall": sum(required) / len(required) if required else 1.0,
        "required_fact_matches": required,
        "forbidden_claim_count": sum(forbidden),
        "forbidden_claim_matches": forbidden,
        "answer_chars": len(answer),
    }


def run_fixture(fixture: FrozenFixture, variant: PromptVariant, *, live: bool = False) -> dict[str, Any]:
    payload = payload_for_variant(fixture, variant)
    system = payload["messages"][0]["content"]
    user = payload["messages"][-1]["content"]
    result: dict[str, Any] = {
        "fixture": fixture.identifier,
        "variant": variant.identifier,
        "model": payload.get("model"),
        "max_output_tokens": payload.get("max_tokens"),
        "generation_prompt_chars": len(system),
        "generation_context_chars": len(fixture.context),
        "generation_total_prompt_chars": sum(len(message.get("content", "")) for message in payload["messages"]),
        "system_prompt": system,
        "user_prompt": user,
        "answer": None,
        "latency_ms": None,
        "token_usage": None,
    }
    if live:
        started = time.perf_counter()
        answer = generate_from_payload(payload)
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        result["answer"] = answer
        result.update(evaluate_answer(fixture, answer))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call the configured final provider; omitted means payload audit only.")
    parser.add_argument("--synthetic", action="store_true", help="Use anonymized fixtures; required for a live experiment.")
    parser.add_argument("--output", help="Write JSON results to this path.")
    args = parser.parse_args()
    if args.live and not args.synthetic:
        parser.error("--live requires --synthetic so real fixtures never leave the workspace")
    fixtures = SYNTHETIC_FIXTURES if args.synthetic else FIXTURES
    results = [run_fixture(fixture, variant, live=args.live) for fixture in fixtures for variant in VARIANTS]
    data = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(data + "\n")
    else:
        print(data)


if __name__ == "__main__":
    main()
