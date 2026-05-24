# -*- coding: utf-8 -*-
"""
Heuristic multi-emotion detection for FR/EN chat text.
Returns a dict: {label, confidence, reasons, scores}
Labels: cheerful, neutral, sad, frustrated, angry, confused, urgent, polite_formal
No external dependencies.
"""
from __future__ import annotations
from typing import List, Dict, Tuple
import re

# Emoji groups
POS_EMOJI = {"😊","😄","😃","🙂","👍","😁","🎉","✨","🤩","🙌","👏","😺"}
SAD_EMOJI = {"😢","😞","😔","☹️","😟","😭","🥺","💔"}
ANGRY_EMOJI = {"😡","😠","🤬"}
CONFUSE_EMOJI = {"🤔","😕","❓","❔"}
URGENT_EMOJI = {"⚠️","⛔","🚨"}

# Keyword lexicons (lowercased)
CHEER_WORDS_FR = {"cool","super","génial","genial","top","nickel","excellent","parfait","trop bien","magnifique","au top","bravo","trop cool","trop fort"}
CHEER_WORDS_EN = {"awesome","great","amazing","fantastic","wonderful","nice","cool","love it","so good","perfect","bravo"}

SAD_WORDS_FR = {"triste","déçu","decu","décevant","decevant","désolé","desole","chagrin"}
SAD_WORDS_EN = {"sad","depressed","unhappy","disappointed","sorry","upset"}

FRUSTRATED_WORDS_FR = {"marche pas","ne marche pas","ne fonctionne pas","fonctionne pas","bug","bloqué","bloque","bloquée","bloquee","galère","galere","n'en peux plus","ras-le-bol","fatiguant","chiant","casse-pieds","grrr","wtf"}
FRUSTRATED_WORDS_EN = {"doesn't work","doesnt work","not working","bug","stuck","blocked","annoying","frustrating","wtf","ugh","argh"}

ANGRY_WORDS_FR = {"colère","colere","furieux","furieuse","énervé","enerve","énervée","enervee","rage","scandale","inacceptable"}
ANGRY_WORDS_EN = {"angry","furious","mad","pissed","rage","outrage","unacceptable"}

CONFUSED_WORDS_FR = {"je ne comprends pas","pas clair","confus","confuse","incompréhensible","incomprehensible","c'est quoi","qu'est-ce que","hein?"}
CONFUSED_WORDS_EN = {"i don't understand","dont understand","not clear","confused","what is","what's","huh?","how does"}

URGENT_WORDS_FR = {"urgent","urgence","rapidement","tout de suite","immédiat","immediat","asap","vite","au plus vite","maintenant"}
URGENT_WORDS_EN = {"urgent","asap","right now","immediately","now","quick","fast","priority"}

POLITE_FORMAL_FR = {"s'il vous plaît","svp","pourriez-vous","serait-il possible","merci d'avance","je vous prie"}
POLITE_FORMAL_EN = {"please","would you","could you","i would appreciate","thank you in advance"}

INSULTS = {"connard","abruti","idiot","stupid","moron","dumb","con"}

def _lc(s: str) -> str:
    return (s or "").strip().lower()

def _count_any(text: str, patterns: List[str] | set[str]) -> int:
    t = _lc(text)
    return sum(1 for p in patterns if p in t)

def _score(text: str, history: List[str]) -> Dict[str, float]:
    """Compute raw scores per mood label from text and short history."""
    all_text = "\n".join([text] + list(history or []))
    t = _lc(all_text)

    # Base counts
    excl = all_text.count("!")
    qmarks = all_text.count("?")

    # Emoji counts
    pos_e = sum(1 for ch in all_text if ch in POS_EMOJI)
    sad_e = sum(1 for ch in all_text if ch in SAD_EMOJI)
    ang_e = sum(1 for ch in all_text if ch in ANGRY_EMOJI)
    conf_e = sum(1 for ch in all_text if ch in CONFUSE_EMOJI)
    urg_e = sum(1 for ch in all_text if ch in URGENT_EMOJI)

    scores = {
        "cheerful": 0.0,
        "sad": 0.0,
        "frustrated": 0.0,
        "angry": 0.0,
        "confused": 0.0,
        "urgent": 0.0,
        "polite_formal": 0.0,
        "neutral": 0.0,
    }

    # Cheerful
    scores["cheerful"] += pos_e * 1.5 + _count_any(t, CHEER_WORDS_FR | CHEER_WORDS_EN) * 1.2
    if excl >= 2:  # excitement
        scores["cheerful"] += 0.8

    # Sad
    scores["sad"] += sad_e * 2.0 + _count_any(t, SAD_WORDS_FR | SAD_WORDS_EN) * 1.5

    # Frustrated
    scores["frustrated"] += _count_any(t, FRUSTRATED_WORDS_FR | FRUSTRATED_WORDS_EN) * 1.3

    # Angry
    scores["angry"] += ang_e * 1.8 + _count_any(t, ANGRY_WORDS_FR | ANGRY_WORDS_EN | INSULTS) * 1.4

    # Confused
    scores["confused"] += conf_e * 1.2 + _count_any(t, CONFUSED_WORDS_FR | CONFUSED_WORDS_EN) * 1.2
    if qmarks >= 2:
        scores["confused"] += 0.6

    # Urgent
    scores["urgent"] += urg_e * 2.0 + _count_any(t, URGENT_WORDS_FR | URGENT_WORDS_EN) * 1.6
    if excl >= 3:
        scores["urgent"] += 0.5

    # Polite formal
    scores["polite_formal"] += _count_any(t, POLITE_FORMAL_FR | POLITE_FORMAL_EN) * 1.2
    if re.search(r"\b(monsieur|madame|cher|chère)\b", t):
        scores["polite_formal"] += 0.6

    # Neutral baseline
    if all(v == 0 for k, v in scores.items() if k != "neutral"):
        scores["neutral"] = 1.0

    return scores

def _top_label(scores: Dict[str, float]) -> Tuple[str, float]:
    label = max(scores.items(), key=lambda kv: kv[1])[0]
    smax = scores[label]
    # normalize confidence between 0.3 and 0.99
    conf = 0.3 + min(0.69, smax / (smax + 3.0))
    return label, round(conf, 2)

def detect_mood(text: str, history_user_msgs: List[str] | None = None) -> Dict:
    """
    Detect user's mood from current text and last few user messages.
    Returns dict: {label, confidence, reasons, scores}
    """
    history_user_msgs = history_user_msgs or []
    scores = _score(text, history_user_msgs[-3:])
    label, conf = _top_label(scores)

    reasons: List[str] = []
    t = _lc("\n".join([text] + history_user_msgs[-3:]))
    if any(e in text for e in POS_EMOJI):
        reasons.append("positive emojis")
    if any(e in text for e in SAD_EMOJI):
        reasons.append("sad emojis")
    if any(e in text for e in ANGRY_EMOJI):
        reasons.append("angry emojis")
    if any(w in t for w in CHEER_WORDS_FR | CHEER_WORDS_EN):
        reasons.append("cheerful keywords")
    if any(w in t for w in SAD_WORDS_FR | SAD_WORDS_EN):
        reasons.append("sad keywords")
    if any(w in t for w in FRUSTRATED_WORDS_FR | FRUSTRATED_WORDS_EN):
        reasons.append("frustration keywords")
    if any(w in t for w in ANGRY_WORDS_FR | ANGRY_WORDS_EN | INSULTS):
        reasons.append("anger keywords")
    if any(w in t for w in URGENT_WORDS_FR | URGENT_WORDS_EN):
        reasons.append("urgent keywords")
    if any(w in t for w in POLITE_FORMAL_FR | POLITE_FORMAL_EN):
        reasons.append("polite/formal markers")
    if text.count('!') >= 2:
        reasons.append("excitement (multiple !)")
    if text.count('?') >= 2:
        reasons.append("confusion (multiple ?)")

    return {"label": label, "confidence": conf, "reasons": reasons, "scores": scores}
