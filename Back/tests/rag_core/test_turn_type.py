from unittest.mock import patch

import numpy as np

from rag_core.turn_type import classify_turn


class _Embedding:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, values, **_kwargs):
        return np.asarray([self.vectors[value] for value in values], dtype=np.float32)


def _classify(text, vector, history=None):
    model = _Embedding({text: vector})
    prototypes = {
        "answer_seeking": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "conversational_continuation": np.asarray([[0.0, 1.0]], dtype=np.float32),
    }
    with patch("rag_core.turn_type._vectors", return_value=prototypes):
        return classify_turn(text, model, history)


def test_incomplete_introductions_are_conversational_continuations():
    for text in (
        "J'ai reçu une question sans autre détail.",
        "Une situation vient de se présenter.",
        "Attendez, je vais préciser le contexte.",
        "Je viens de recevoir un message inhabituel.",
        "On m'a parlé d'un sujet.",
    ):
        result = _classify(text, [0.0, 1.0])
        assert result.turn_type == "conversational_continuation"
        assert result.decision_reason == "semantic_incomplete"


def test_actionable_requests_are_answer_seeking_with_or_without_question_mark():
    for text, vector in (
        ("Explique-moi la cause de cette fuite.", [1.0, 0.0]),
        ("Trouve les documents liés à cette référence.", [1.0, 0.0]),
        ("Je cherche la documentation de la référence AX-17", [1.0, 0.0]),
        ("Peut-on utiliser cet équipement à basse température ?", [0.0, 1.0]),
    ):
        result = _classify(text, vector)
        assert result.turn_type == "answer_seeking"


def test_detailed_statement_is_actionable_without_making_every_statement_actionable():
    detailed = _classify("La référence AX-17 fuit depuis plusieurs heures à chaud", [0.0, 1.0])
    incomplete = _classify("J'ai une difficulté.", [0.0, 1.0])
    assert detailed.turn_type == "answer_seeking"
    assert detailed.decision_reason == "detailed_statement"
    assert incomplete.turn_type == "conversational_continuation"


def test_follow_up_question_stays_answer_seeking_with_history():
    result = _classify(
        "Est-ce possible dans ce cas ?", [0.0, 1.0],
        history=[
            {"role": "user", "content": "J'ai reçu une question sans autre détail."},
            {"role": "assistant", "content": "Quelle est la question ?"},
        ],
    )
    assert result.turn_type == "answer_seeking"
    assert result.decision_reason == "explicit_question"
