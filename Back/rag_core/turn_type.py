"""Lightweight semantic classification for the role of the current user turn."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

TurnType = Literal["answer_seeking", "conversational_continuation"]

# Conceptual anchors, not domain examples or keyword rules.  The multilingual
# embedding model compares the meaning of a turn to these two discourse roles.
_TURN_PROTOTYPES = {
    "answer_seeking": (
        "L'utilisateur formule une demande précise qui peut être traitée maintenant.",
        "La phrase contient assez d'information pour répondre, chercher ou agir.",
        "The user makes a complete request that can be handled now.",
    ),
    "conversational_continuation": (
        "L'utilisateur introduit une situation sans avoir encore formulé de demande précise.",
        "Le message apporte un préambule et attend une poursuite de la conversation.",
        "The user gives context but has not yet stated what help is needed.",
    ),
}
_CACHE: dict[int, dict[str, np.ndarray]] = {}


@dataclass(frozen=True)
class TurnClassification:
    turn_type: TurnType
    semantic_margin: float | None
    decision_reason: str


def _vectors(embed_model):
    cache_key = id(embed_model)
    if cache_key not in _CACHE:
        _CACHE[cache_key] = {
            label: np.asarray(embed_model.encode(list(items), convert_to_tensor=False, normalize_embeddings=True), dtype=np.float32)
            for label, items in _TURN_PROTOTYPES.items()
        }
    return _CACHE[cache_key]


def _has_structural_detail(q: str) -> bool:
    """A domain-neutral cue for a detailed, actionable statement without ``?``."""
    tokens = [token for token in q.split() if token]
    return len(tokens) >= 7 and any(character.isdigit() for character in q)


def classify_turn(q: str, embed_model, history: Sequence[dict] | None = None, *, threshold: float = 0.015) -> TurnClassification:
    """Classify whether a user has supplied enough information to act now.

    Embedding similarity supplies the semantic decision.  Punctuation and
    structural detail are only confidence signals, so no vocabulary or domain
    pattern determines the outcome.
    """
    if not q or not q.strip():
        return TurnClassification("answer_seeking", None, "empty_fallback")
    text = q.strip()
    if "?" in text:
        return TurnClassification("answer_seeking", None, "explicit_question")
    if _has_structural_detail(text):
        return TurnClassification("answer_seeking", None, "detailed_statement")
    try:
        query = np.asarray(
            embed_model.encode([text], convert_to_tensor=False, normalize_embeddings=True)[0],
            dtype=np.float32,
        )
        scores = {label: float(np.max(matrix @ query)) for label, matrix in _vectors(embed_model).items()}
        margin = scores["conversational_continuation"] - scores["answer_seeking"]
        if margin >= threshold:
            return TurnClassification("conversational_continuation", margin, "semantic_incomplete")
        return TurnClassification("answer_seeking", margin, "semantic_actionable_or_ambiguous")
    except Exception:
        return TurnClassification("answer_seeking", None, "embedding_fallback")


def classify_turn_type(q: str, embed_model, history: Sequence[dict] | None = None, *, threshold: float = 0.015) -> TurnType:
    """Compatibility wrapper used by the pipeline."""
    return classify_turn(q, embed_model, history, threshold=threshold).turn_type
