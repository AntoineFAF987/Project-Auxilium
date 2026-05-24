# -*- coding: utf-8 -*-
"""
Vérification de faithfulness (fidélité) des réponses par rapport au contexte.
Utilise un modèle NLI léger pour scorer si la réponse est entailed par le contexte.
"""
from typing import Optional, Dict
import re

# Chargement lazy du modèle NLI
_NLI_MODEL = None
_NLI_AVAILABLE = None

NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
NLI_MODEL_FALLBACK = "cross-encoder/nli-deberta-v3-small"  # EN uniquement, plus léger


def _load_nli_model():
    """Charge le modèle NLI multilingue (lazy loading)."""
    global _NLI_MODEL, _NLI_AVAILABLE
    if _NLI_AVAILABLE is not None:
        return _NLI_MODEL
    
    try:
        from transformers import pipeline
        _NLI_MODEL = pipeline(
            "text-classification",
            model=NLI_MODEL_NAME,
            device=-1,  # CPU pour éviter saturation GPU
        )
        _NLI_AVAILABLE = True
        print(f"✅ Modèle NLI multilingue chargé: {NLI_MODEL_NAME}")
        return _NLI_MODEL
    except Exception as e:
        print(f"⚠️  NLI multilingue indisponible ({e}), fallback EN...")
        try:
            from transformers import pipeline
            _NLI_MODEL = pipeline(
                "text-classification",
                model=NLI_MODEL_FALLBACK,
                device=-1,
            )
            _NLI_AVAILABLE = True
            print(f"✅ Modèle NLI fallback chargé: {NLI_MODEL_FALLBACK}")
            return _NLI_MODEL
        except Exception as e2:
            print(f"ℹ️  Aucun modèle NLI disponible ({e2}). Faithfulness check désactivé.")
            _NLI_AVAILABLE = False
            return None


def check_faithfulness(answer: str, context: str, threshold: float = 0.5) -> Dict:
    """
    Vérifie si la réponse est fidèle au contexte (entailed).
    
    Args:
        answer: La réponse générée par le LLM
        context: Le contexte fourni au LLM
        threshold: Seuil de confiance pour considérer entailed (0.5 par défaut)
    
    Returns:
        Dict avec keys: faithful (bool), score (float), label (str), reason (str)
    """
    if not context or not context.strip():
        return {"faithful": True, "score": 1.0, "label": "no_context", "reason": "Pas de contexte strict"}
    
    if not answer or not answer.strip():
        return {"faithful": False, "score": 0.0, "label": "empty_answer", "reason": "Réponse vide"}
    
    # Nettoyer la réponse (enlever les citations markup)
    answer_clean = re.sub(r'<CITATIONS>.*?</CITATIONS>', '', answer, flags=re.IGNORECASE).strip()
    answer_clean = re.sub(r'\[WEB\].*?\[/WEB\]', '', answer_clean, flags=re.IGNORECASE).strip()
    
    # Tronquer si trop long (limite du modèle NLI ~512 tokens)
    if len(answer_clean) > 2000:
        answer_clean = answer_clean[:2000]
    if len(context) > 4000:
        context = context[:4000]
    
    model = _load_nli_model()
    if model is None:
        # Fallback: pas de vérification
        return {"faithful": True, "score": 1.0, "label": "nli_unavailable", "reason": "NLI non disponible"}
    
    try:
        # Format NLI: premise=context, hypothesis=answer
        result = model(f"{context} [SEP] {answer_clean}", truncation=True)
        
        # result peut être: [{"label": "ENTAILMENT", "score": 0.95}, ...]
        label = result[0]["label"].upper() if isinstance(result, list) else result["label"].upper()
        score = float(result[0]["score"]) if isinstance(result, list) else float(result["score"])
        
        # Labels possibles: ENTAILMENT, NEUTRAL, CONTRADICTION
        if "ENTAIL" in label:
            faithful = score >= threshold
            reason = f"Entailed (score={score:.2f})"
        elif "CONTRADICTION" in label:
            faithful = False
            reason = f"Contradictoire (score={score:.2f})"
        else:  # NEUTRAL
            faithful = score >= (threshold + 0.1)  # seuil plus strict pour neutral
            reason = f"Neutre (score={score:.2f})"
        
        return {"faithful": faithful, "score": score, "label": label, "reason": reason}
    
    except Exception as e:
        print(f"⚠️  Erreur NLI check: {e}")
        return {"faithful": True, "score": 1.0, "label": "error", "reason": f"Erreur: {e}"}


def should_fallback_to_general(faithfulness_check: Dict, strict_mode: bool) -> bool:
    """
    Détermine si on doit basculer en mode GENERAL suite au check de faithfulness.
    
    Args:
        faithfulness_check: Résultat de check_faithfulness()
        strict_mode: Si True, on était en mode strict avec contexte local
    
    Returns:
        True si on doit redemander en mode GENERAL
    """
    if not strict_mode:
        return False  # déjà en mode général, pas de fallback
    
    if faithfulness_check["label"] in {"no_context", "nli_unavailable", "error"}:
        return False  # pas de vérification possible ou pas de contexte
    
    # Si non-faithful ET score bas => probablement hallucination
    if not faithfulness_check["faithful"] and faithfulness_check["score"] < 0.4:
        return True
    
    return False
