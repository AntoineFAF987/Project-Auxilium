"""Deterministic query variants and rank fusion before CandidatePool creation."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


def _canonical(value: str) -> str:
    text = unicodedata.normalize("NFKD", value.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _obviously_duplicate(left: str, right: str) -> bool:
    """Deduplicate only safe textual equivalents; preserve useful rewrites."""
    left_canonical, right_canonical = _canonical(left), _canonical(right)
    if left_canonical == right_canonical:
        return True
    return bool(left_canonical and set(left_canonical.split()) == set(right_canonical.split()))


def _specific_anchors(text: str) -> set[str]:
    """Return exact, domain-neutral identifiers and uppercase reference groups.

    These are deliberately lexical rather than product-specific: model names,
    dimensions, versions and all-caps reference groups must survive a query
    expansion byte-for-byte (apart from harmless surrounding whitespace).
    """
    anchors = set(re.findall(
        r"\b(?:[A-Za-z]+[A-Za-z0-9-]*\d[A-Za-z0-9-]*|\d+(?:\.\d+)+|\d{2,})\b",
        text or "",
    ))
    anchors.update(re.findall(r"\b[A-Z]{2,}(?:[ -][A-Z0-9]{2,})*\b", text or ""))
    return anchors


_FRENCH_MARKERS = {
    "quel", "quelle", "quels", "quelles", "est", "sont", "le", "la", "les",
    "du", "des", "pour", "avec", "dans", "cherche", "encore", "anglais", "et",
    "temperature", "pression", "course",
    "vanne", "actionneur", "etancheite", "certification", "limite", "plage",
}
_ENGLISH_MARKERS = {
    "what", "which", "is", "are", "the", "of", "for", "with", "in", "actuator",
    "valve", "pressure", "temperature", "dead", "band", "certification", "range",
}


def detect_query_language(query: str) -> str:
    """Classify only the languages supported by the first expansion release."""
    tokens = set(_canonical(query).split())
    french = len(tokens & _FRENCH_MARKERS)
    english = len(tokens & _ENGLISH_MARKERS)
    # Diacritics are a useful tie breaker for short French technical queries.
    has_french_diacritic = bool(re.search(r"[àâçéèêëîïôùûüÿœ]", query.casefold()))
    if french > english or (french and has_french_diacritic):
        return "fr"
    if english > french:
        return "en"
    return "unknown"


# A compact vocabulary of common engineering concepts, not a catalogue or a
# product-specific glossary.  It intentionally emits search terms, never a
# natural-language sentence or a free translation.
_FR_DOCUMENTARY_TERMS: tuple[tuple[str, str], ...] = (
    ("bande morte", "dead band"),
    ("plage de reglage", "adjustment range"),
    ("temperature maximale", "maximum temperature"),
    ("temperature max", "maximum temperature"),
    ("pression maximale", "maximum pressure"),
    ("pression max", "maximum pressure"),
    ("etancheite", "tightness"),
    ("certification", "certification"),
    ("materiau", "material"),
    ("actionneurs", "actuator"),
    ("actionneur", "actuator"),
    ("vannes", "valve"),
    ("vanne", "valve"),
    ("couple", "torque"),
    ("course", "stroke"),
    ("pression", "pressure"),
    ("temperature", "temperature"),
    ("limite", "limit"),
    ("plage", "range"),
)


@dataclass(frozen=True)
class CrossLanguageQueryDecision:
    query: str | None
    target_language: str | None
    rejection_reason: str | None = None
    source_query: str | None = None
    information_need: str | None = None
    information_need_retained: bool = False
    semantic_intent_retained: bool = False
    validation_passed: bool = False
    validation_reasons: tuple[str, ...] = ()
    query_before_validation: str | None = None
    query_after_validation: str | None = None


_GENERIC_ENTITY_TERMS = frozenset({"actuator", "valve", "positioner", "device", "product", "overview"})
_SEMANTIC_INTENT_TERMS = {
    "procedure": frozenset({"adjustment", "setting", "procedure", "configuration", "configure", "installation", "mounting"}),
    "decision": frozenset({"decision", "approval", "status", "accepted", "rejected"}),
    "current_state": frozenset({"current", "latest", "status", "state"}),
    "comparison": frozenset({"comparison", "difference", "versus"}),
}


def _anchor_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def validate_cross_language_query(
    *, candidate: str, anchors: set[str], query_semantics: str | None,
) -> CrossLanguageQueryDecision:
    """Reject a translated search query that retained only its product subject."""
    candidate = candidate.strip()[:240]
    candidate_anchors = {_anchor_key(value) for value in _specific_anchors(candidate)}
    required_anchors = {_anchor_key(value) for value in anchors}
    anchors_preserved = required_anchors.issubset(candidate_anchors)
    terms = set(_canonical(candidate).split())
    anchor_terms = set()
    for anchor in anchors:
        anchor_terms.update(_canonical(anchor).split())
        anchor_terms.add(_anchor_key(anchor))
    need_terms = terms - anchor_terms - _GENERIC_ENTITY_TERMS
    semantic_terms = _SEMANTIC_INTENT_TERMS.get(query_semantics or "", frozenset())
    semantic_retained = not semantic_terms or bool(terms & semantic_terms)
    # Intent words are not themselves a property; a procedure still needs a
    # documentary target in addition to "procedure" or "adjustment".
    information_terms = need_terms - semantic_terms
    information_retained = bool(information_terms) or (
        query_semantics in {"procedure", "decision", "current_state"} and bool(terms & semantic_terms)
    )
    reasons: list[str] = []
    if not anchors_preserved:
        reasons.append("anchors_not_preserved")
    if not information_retained:
        reasons.append("information_need_lost")
    if not semantic_retained:
        reasons.append("semantic_intent_lost")
    return CrossLanguageQueryDecision(
        candidate if not reasons else None, "en", reasons[0] if reasons else None,
        information_need=" ".join(sorted(information_terms)) or None,
        information_need_retained=information_retained,
        semantic_intent_retained=semantic_retained,
        validation_passed=not reasons,
        validation_reasons=tuple(reasons),
        query_before_validation=candidate,
        query_after_validation=candidate if not reasons else None,
    )


def evaluate_cross_language_query(
    *, original_user_query: str, orchestrator_query: str | None,
    detected_language: str, anchors: set[str] | None = None,
    query_semantics: str | None = "general_document_question",
    proposed_query: str | None = None,
) -> CrossLanguageQueryDecision:
    """Build at most one conservative French-to-English documentary variant."""
    if detected_language != "fr":
        return CrossLanguageQueryDecision(None, None, "source_language_not_french")
    if query_semantics is None:
        return CrossLanguageQueryDecision(None, "en", "non_documentary_query")

    source = " ".join(part for part in (orchestrator_query, original_user_query) if part).strip()
    exact_anchors = anchors if anchors is not None else _specific_anchors(source)
    if proposed_query:
        validated = validate_cross_language_query(
            candidate=proposed_query, anchors=exact_anchors, query_semantics=query_semantics,
        )
        return CrossLanguageQueryDecision(
            **{**validated.__dict__, "source_query": orchestrator_query or original_user_query}
        )
    canonical_source = _canonical(source)
    translated: list[str] = []
    for french, english in _FR_DOCUMENTARY_TERMS:
        if french in canonical_source and english not in translated:
            translated.append(english)
    if not translated:
        return CrossLanguageQueryDecision(None, "en", "no_supported_documentary_terms", source_query=orchestrator_query or original_user_query)

    # The union makes planner rewrites additive: an anchor present only in the
    # raw user wording cannot disappear from the English documentary query.
    ordered_anchors = [anchor for anchor in _specific_anchors(source) if anchor in exact_anchors]
    for anchor in sorted(exact_anchors, key=lambda value: (value.casefold(), value)):
        if anchor not in ordered_anchors:
            ordered_anchors.append(anchor)
    # Search headings conventionally put the equipment/entity before the
    # requested property ("actuator dead band", not a translated sentence).
    entity_terms = {"actuator", "valve"}
    translated.sort(key=lambda term: (term not in entity_terms,))
    if query_semantics == "procedure" and "adjustment" not in translated:
        translated.append("adjustment")
    candidate = " ".join([*ordered_anchors, *translated]).strip()
    if len(candidate) < 3:
        return CrossLanguageQueryDecision(None, "en", "empty_cross_language_query", source_query=orchestrator_query or original_user_query)
    if _obviously_duplicate(candidate, original_user_query) or (
        orchestrator_query and _obviously_duplicate(candidate, orchestrator_query)
    ):
        return CrossLanguageQueryDecision(None, "en", "duplicate_existing_query", source_query=orchestrator_query or original_user_query)
    validated = validate_cross_language_query(candidate=candidate, anchors=exact_anchors, query_semantics=query_semantics)
    return CrossLanguageQueryDecision(**{**validated.__dict__, "source_query": orchestrator_query or original_user_query})


def build_cross_language_query(**kwargs: Any) -> str | None:
    """Public convenience API for callers that only need the optional query."""
    return evaluate_cross_language_query(**kwargs).query


def normalized_query(original: str) -> str | None:
    """Remove conversational glue only; never add or substitute domain terms."""
    filler = {
        "je", "me", "m", "demandais", "demande", "si", "j", "ai", "avais", "enfin",
        "peux", "tu", "vous", "pourrais", "voudrais", "savoir", "juste", "svp", "please",
        "can", "you", "could", "would", "i", "do", "does", "the", "a", "an",
    }
    words = re.findall(r"[\wÀ-ÿ0-9-]+", original)
    compact = [word for word in words if _canonical(word) not in filler]
    candidate = " ".join(compact).strip()
    if len(compact) < 3 or _canonical(candidate) == _canonical(original):
        return None
    return candidate[:400]


def _followup_additions(raw_user_message: str, subject: str) -> str:
    """Keep only structured documentary refinements from a follow-up.

    The orchestrator has already resolved the subject. Free prose in a
    follow-up is an instruction to the assistant, not a search query. We
    retain generic structural refinements only: reference-like identifiers,
    a named entity, and an explicit email/date locator.
    """
    subject_terms = set(_canonical(subject).split())
    words = re.findall(r"[\wÀ-ÿ0-9-]+", raw_user_message)
    additions: list[str] = []
    for index, word in enumerate(words):
        canonical = _canonical(word)
        if not canonical or canonical in subject_terms:
            continue
        if any(char.isdigit() for char in word):
            additions.append(word)
        elif index > 0 and re.fullmatch(r"[A-ZÀ-Ö][A-Za-zÀ-ÿ'-]{2,}", word):
            additions.append(word)
    lowered = [_canonical(word) for word in words]
    for index, token in enumerate(lowered):
        if token not in {"mail", "email", "courriel"}:
            continue
        window = words[max(0, index - 2):index + 4]
        if any(any(char.isdigit() for char in value) for value in window):
            additions.extend(value for value in window if any(char.isdigit() for char in value))
            additions.append(words[index])
    return " ".join(dict.fromkeys(additions))


def resolve_retrieval_query(*, raw_user_message: str, orchestrator_query: str, history: list[dict[str, Any]] | None = None) -> str:
    """Resolve a follow-up to its subject without a second model invocation.

    The orchestrator's standalone subject is the stable base; the current turn
    can only append explicit, non-conversational qualifiers (for example a name).
    History is intentionally accepted as execution context, while subject
    resolution stays deterministic and does not reinterpret free-form prose.
    """
    del history  # The plan was built from compact history; avoid a second semantic pass here.
    subject = orchestrator_query.strip()
    additions = _followup_additions(raw_user_message, subject)
    return " ".join(part for part in (subject, additions) if part).strip()


def build_retrieval_queries(*, original_query: str, orchestrator_query: str | None, resolved_query: str | None = None, follow_up: bool = False, cross_language_query: str | None = None) -> list[tuple[str, str]]:
    """Build factual query variants independently of any response presentation.

    An autonomous user query is retained because it can contain exact anchors
    lost by a rewrite. For a resolved conversational follow-up, the raw turn
    is preserved in history and logs but is not itself a documentary query.
    """
    # The user's wording is evidence too: it often contains an exact model,
    # reference, or relation that a planner paraphrase can accidentally lose.
    # A resolved follow-up is additive, never a silent replacement for it.
    standalone = resolved_query or orchestrator_query
    options: list[tuple[str, str | None]] = (
        [
            ("resolved_followup", standalone),
            ("normalized", normalized_query(standalone or original_query)),
        ]
        if follow_up else [
            ("original_autonomous", original_query),
            ("orchestrator", standalone),
            ("normalized", normalized_query(standalone or original_query)),
        ]
    )
    if cross_language_query:
        options.append(("cross_language", cross_language_query))
    result: list[tuple[str, str]] = []
    for kind, query in options:
        if not query or len(query.strip()) < 3:
            continue
        if any(_obviously_duplicate(query, existing) for _, existing in result):
            continue
        result.append((kind, query.strip()))
    return result[:4]


def reciprocal_rank_fusion(
    rankings: list[tuple[str, list[tuple[float, dict[str, Any]]]]], *, k: int = 60,
) -> list[tuple[float, dict[str, Any]]]:
    """Fuse query-specific rankings without comparing their raw score scales."""
    merged: dict[str, tuple[float, dict[str, Any], dict[str, int], dict[str, float]]] = {}
    for query_kind, rows in rankings:
        for rank, (score, meta) in enumerate(rows, start=1):
            uid = str(meta.get("chunk_uid") or f"{meta.get('document_id')}:{meta.get('chunk_id')}")
            fusion, best_meta, ranks, scores = merged.get(uid, (0.0, dict(meta), {}, {}))
            ranks[query_kind] = rank
            scores[query_kind] = float(score)
            merged[uid] = (fusion + 1.0 / (k + rank), best_meta, ranks, scores)
    result = []
    for fusion, meta, ranks, scores in merged.values():
        enriched = dict(meta)
        enriched["retrieved_by"] = list(ranks)
        enriched["per_query_rank"] = ranks
        enriched["per_query_score"] = scores
        enriched["fusion_score"] = fusion
        result.append((fusion, enriched))
    return sorted(result, key=lambda item: item[0], reverse=True)
