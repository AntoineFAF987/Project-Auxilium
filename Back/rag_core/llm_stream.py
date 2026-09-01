# -*- coding: utf-8 -*-
"""
ask_mistral_with_context_stream — Version streaming pour affichage progressif
"""
from typing import List, Dict, Iterator, Optional
from .affect import detect_mood
from .capability_contract import LOCAL_SOURCE_CAPABILITY_EN, LOCAL_SOURCE_CAPABILITY_FR
from runtime_settings import get_runtime_settings
from .llm_providers import stream_from_payload


def ask_mistral_with_context_stream(
    question: str,
    context_text: str,
    history: List[Dict] = None,
    model: str = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    smalltalk_mode: bool = False,
    smalltalk_kind: str | None = None,
    roleplay_mode: bool = False,
    conversational_mode: bool = False,
    evidence_mode: str = "none",
    **kwargs,
) -> Iterator[str]:
    """
    Version streaming de ask_mistral_with_context.
    Yield des chunks de texte au fur et à mesure.
    """
    import os as _os, requests as _requests

    generation = get_runtime_settings().generation

    api_key = _os.getenv("MISTRAL_API_KEY")
    if generation.provider == "mistral" and not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY absente. Définis-la puis relance l'API.\n"
            'PowerShell :  setx MISTRAL_API_KEY "ta_cle"'
        )

    # Détection FR/EN simple
    def _detect_lang_chat(text: str, prev: str | None = None) -> str:
        t = (text or "").strip().lower()
        if not t: return prev or "fr"
        en_kw = ["hello","hi ","hey","what","why","when","where","which","please","thanks","thank you"]
        fr_kw = ["bonjour","salut","bonsoir","pourquoi","quand","où","comment","svp","s'il te plaît","merci"]
        en_hits = sum(k in t for k in en_kw); fr_hits = sum(k in t for k in fr_kw)
        import re as _re2
        fr_tokens = _re2.findall(r"\b(je|tu|il|elle|on|nous|vous|ils|elles|le|la|les|un|une|des|est|êtes|sont)\b", t)
        en_tokens = _re2.findall(r"\b(i|you|he|she|we|they|the|a|an|is|are|to|of|and)\b", t)
        fr_score = fr_hits + 0.6*len(fr_tokens) + (1 if any(c in "éèêàùûôîïç" for c in t) else 0)
        en_score = en_hits + 0.6*len(en_tokens)
        if abs(fr_score - en_score) < 0.6 and prev in {"fr","en"}: return prev
        return "fr" if fr_score >= en_score else "en"

    prev_lang = None
    if history:
        for m in reversed(history):
            if m.get("role") == "user" and m.get("content"):
                prev_lang = _detect_lang_chat(m["content"]); break
    lang = _detect_lang_chat(question, prev_lang); is_fr = (lang == "fr")

    qlow = (question or "").lower()
    is_casual = (("tu " in qlow or "toi " in qlow) if is_fr else any(k in qlow for k in ["bro","buddy","mate","dude"]))
    is_formal = (("vous" in qlow or "svp" in qlow) if is_fr else any(k in qlow for k in [" please","sir","madam"]))

    # Détection de l'humeur multi-émotions (FR/EN)
    hist_user_msgs = [m.get("content", "") for m in (history or []) if m.get("role") == "user"]
    _mood = detect_mood(question, hist_user_msgs)
    mood_label = (_mood.get("label") or "neutral").lower()
    is_cheerful = (mood_label == "cheerful")
    is_sad = (mood_label == "sad")
    is_frustrated = (mood_label == "frustrated")
    is_angry = (mood_label == "angry")
    is_confused = (mood_label == "confused")
    is_urgent = (mood_label == "urgent")
    is_polite_formal = (mood_label == "polite_formal")

    # Petite fonction utilitaire pour générer des consignes de style selon l'humeur
    def _style_rules() -> List[str]:
        r: List[str] = []
        # Global emoji policy based on mood/formality
        allow_emojis = not (is_polite_formal or is_formal or is_angry or is_frustrated or is_urgent)
        if is_fr:
            if is_cheerful:
                if allow_emojis:
                    r += ["Adopte un ton enthousiaste et chaleureux.", "Tu peux utiliser 1–2 emojis adaptés (pas excessifs). 😊✨"]
                else:
                    r += ["Adopte un ton enthousiaste et chaleureux.", "N'utilise pas d'emojis dans ce contexte."]
            if is_sad:
                if allow_emojis:
                    r += ["Sois empathique et rassurant, parle avec douceur.", "Évite les emojis festifs ; un emoji discret est acceptable au besoin (🌷)."]
                else:
                    r += ["Sois empathique et rassurant, parle avec douceur.", "N'utilise pas d'emojis."]
            if is_frustrated:
                r += ["Reconnais la frustration et va droit au but avec des étapes concrètes.", "Pas d'emojis, ton calme et aidant."]
            if is_angry:
                r += ["Désamorce avec un ton respectueux et posé ; formule une courte excuse si pertinent.", "Pas d'emojis, reste factuel et utile."]
            if is_confused:
                r += ["Clarifie simplement, propose 1–3 options ou une question de clarification.", "Évite le jargon ; un pas-à-pas court est préférable."]
            if is_urgent:
                r += ["Priorise l'action : donne des étapes concises et immédiates.", "Pas d'emojis ni de digressions."]
            if is_polite_formal or is_formal:
                r += ["Maintiens un ton professionnel et poli (vous).", "N'utilise pas d'emojis."]
            if not r:
                if allow_emojis:
                    r += ["Reste naturel, positif et professionnel selon le contexte.", "Un emoji discret est acceptable si pertinent."]
                else:
                    r += ["Reste naturel, positif et professionnel selon le contexte.", "N'utilise pas d'emojis."]
        else:
            if is_cheerful:
                if allow_emojis:
                    r += ["Adopt an enthusiastic, warm tone.", "You may use 1–2 appropriate emojis (not excessive). 😊✨"]
                else:
                    r += ["Adopt an enthusiastic, warm tone.", "Do not use emojis in this context."]
            if is_sad:
                if allow_emojis:
                    r += ["Be empathetic and reassuring, with a gentle tone.", "Avoid festive emojis; a soft emoji is acceptable if needed (🌷)."]
                else:
                    r += ["Be empathetic and reassuring, with a gentle tone.", "Do not use emojis."]
            if is_frustrated:
                r += ["Acknowledge frustration and get straight to actionable steps.", "No emojis; keep a calm, helpful tone."]
            if is_angry:
                r += ["De-escalate respectfully; a brief apology if relevant.", "No emojis; be factual and helpful."]
            if is_confused:
                r += ["Clarify simply; offer 1–3 options or a clarifying question.", "Avoid jargon; prefer a short step-by-step."]
            if is_urgent:
                r += ["Prioritize action: concise, immediate steps.", "No emojis or digressions."]
            if is_polite_formal or is_formal:
                r += ["Maintain a professional, polite tone.", "Do not use emojis."]
            if not r:
                if allow_emojis:
                    r += ["Keep it natural, positive, and professional as appropriate.", "A subtle emoji is acceptable if appropriate."]
                else:
                    r += ["Keep it natural, positive, and professional as appropriate.", "Do not use emojis."]
        return r

    def _adjust_temp(base: float) -> float:
        # Baisse la créativité pour colère/urgence/confusion
        if is_angry or is_frustrated or is_urgent:
            return min(base, 0.55)
        if is_confused:
            return min(base, 0.6)
        if is_cheerful:
            return max(base, 0.65)
        if is_sad:
            return max(min(base, 0.65), 0.55)
        return base

    messages: List[Dict] = []
    if conversational_mode:
        messages = [{"role": "system", "content": (
            "The user is introducing context, not asking a complete factual question yet. "
            "Reply briefly and naturally, acknowledge the situation, and ask one helpful clarifying question. "
            "Do not search for or invent a documentary answer."
        )}]
        messages.extend({"role": m["role"], "content": m["content"]} for m in (history or []) if isinstance(m, dict) and "role" in m and "content" in m)
        messages.append({"role": "user", "content": question})
        yield from stream_from_payload({"model": model or generation.model, "messages": messages, "temperature": temperature if temperature is not None else generation.smalltalk_temperature, "top_p": generation.top_p, "max_tokens": max_tokens or 160, "stream": True})
        return
    NO_META_FR = "N'emploie jamais de phrases méta (ex. « je suis une IA »)."
    NO_META_EN = "Never use meta statements (e.g., 'I am an AI')."

    # 1) ROLEPLAY
    if roleplay_mode:
        sys_base = "Tu es une IA chaleureuse et empathique. " if is_fr else "You are a warm and empathetic AI. "
        sys = (sys_base + "Tu peux SIMULER une situation quand on te le demande. " + NO_META_FR) if is_fr else (sys_base + "You may ROLEPLAY when asked. " + NO_META_EN)
        style = " ".join(_style_rules())
        if style:
            sys += " " + style
        messages.append({"role": "system", "content": sys})
        for m in (history or []):
            if isinstance(m, dict) and "role" in m and "content" in m:
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": question})
        payload = {
            "model": model or generation.model,
            "messages": messages,
            "temperature": _adjust_temp(max(temperature if temperature is not None else generation.roleplay_temperature, generation.roleplay_temperature)),
            "top_p": generation.top_p,
            "max_tokens": min(max_tokens if max_tokens is not None else generation.roleplay_max_tokens, generation.roleplay_max_tokens),
            "stream": True  # STREAMING ACTIVÉ
        }
        yield from stream_from_payload(payload)
        return

    # 2) SMALL TALK
    if smalltalk_mode:
        rules = []
        if is_fr:
            rules += ["Réponds en français.", "Ton chaleureux, amical et naturel (1–2 phrases).", NO_META_FR]
            if is_casual: rules.append("Tu peux tutoyer si l'utilisateur l'emploie.")
            if is_formal: rules.append("Reste poli (vous).")
            rules += _style_rules()
        else:
            rules += ["Reply in English.", "Warm, friendly, natural tone (1–2 sentences).", NO_META_EN]
            rules += _style_rules()
        messages.append({"role": "system", "content": " ".join(rules)})
        for m in (history or []):
            if isinstance(m, dict) and "role" in m and "content" in m:
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": question})
        payload = {
            "model": model or generation.model,
            "messages": messages,
            "temperature": _adjust_temp(max(temperature if temperature is not None else generation.smalltalk_temperature, generation.smalltalk_temperature)),
            "top_p": generation.top_p,
            "max_tokens": min(max_tokens if max_tokens is not None else generation.smalltalk_max_tokens, generation.smalltalk_max_tokens),
            "stream": True  # STREAMING ACTIVÉ
        }
        yield from stream_from_payload(payload)
        return

    # 3) Mode normal — naturel & strict si contexte
    rules = []
    if is_fr:
        rules += [
            "Réponds en français.", 
            NO_META_FR,
            "Fournis des réponses COMPLÈTES et DÉTAILLÉES.",
            "Structure ta réponse de façon claire et fluide, avec des paragraphes bien distincts.",
            "Utilise le **gras** pour les termes clés importants.",
            "Utilise des listes à puces (•) pour énumérer des éléments, étapes ou caractéristiques.",
            "N'utilise JAMAIS de titres markdown (###, ##, etc.) - reste en texte fluide avec des paragraphes.",
            "Si tu proposes d'approfondir un point, fais-le de manière naturelle et conversationnelle (ex: 'Je peux te détailler X si tu veux' ou 'Dis-moi si tu souhaites que j'approfondisse Y').",
            "Garde un ton chaleureux, naturel et empathique tout en étant informatif et précis."
        ]
        rules += _style_rules()
        if not context_text or not context_text.strip():
            rules.append(LOCAL_SOURCE_CAPABILITY_FR)
        else:
            rules += [
                "Utilise UNIQUEMENT le CONTEXTE fourni ; n'invente jamais une conclusion ou un fait absent.",
                "N'AJOUTE JAMAIS d'informations, comparaisons ou exemples qui ne sont pas explicitement dans le CONTEXTE.",
                "Lorsqu'un CONTEXTE utile est fourni, ne réponds jamais uniquement par une abstention. Restitue d'abord tous les faits pertinents établis par les sources, y compris les étapes, dates, positions, contradictions ou éléments partiels.",
                "Si le CONTEXTE ne permet pas de confirmer la conclusion finale, explique précisément ce qui est établi, ce qui manque pour conclure et formule une conclusion prudente. Pour une évolution dans le temps, synthétise les éléments connus dans leur ordre chronologique.",
                "Même en mode strict, fournis une réponse complète et structurée basée sur le contexte.",
                ("À la toute fin de ta réponse, ajoute une ligne <CITATIONS>[i1,i2,...]</CITATIONS> "
                 "où i1,i2,... sont les numéros [1..N] des sources DU CONTEXTE réellement utilisées. "
                 "N'invente pas de numéros. Si aucune source n'a été nécessaire, écris <CITATIONS>[]</CITATIONS>.")
            ]
    else:
        rules += [
            "Reply in English.", 
            NO_META_EN,
            "Provide COMPLETE and DETAILED answers.",
            "Structure your response clearly with well-defined paragraphs.",
            "Use **bold** for important key terms.",
            "Use bullet lists (•) to enumerate elements, steps, or characteristics.",
            "NEVER use markdown headings (###, ##, etc.) - keep it in flowing text with paragraphs.",
            "If you offer to go deeper on a topic, do it naturally and conversationally (e.g., 'I can detail X if you'd like' or 'Let me know if you want me to expand on Y').",
            "Keep a warm, natural, and empathetic tone while being informative and precise."
        ]
        rules += _style_rules()
        if not context_text or not context_text.strip():
            rules.append(LOCAL_SOURCE_CAPABILITY_EN)
        else:
            rules += [
                "Answer ONLY from the provided CONTEXT; never invent a conclusion or fact that is absent.",
                "NEVER add information, comparisons, or examples that are not explicitly in the CONTEXT.",
                "When useful CONTEXT is provided, never answer with a bare abstention. First state every relevant fact established by the sources, including steps, dates, positions, contradictions, or partial evidence.",
                "If the CONTEXT cannot confirm the final conclusion, explain precisely what is established, what is missing to conclude, and give a cautious conclusion. For changes over time, summarize the known evidence chronologically.",
                "Even in strict mode, provide a complete and structured answer based on context.",
                ("At the very end of your answer, add a line <CITATIONS>[i1,i2,...]</CITATIONS> "
                 "where i1,i2,... are the [1..N] indices of CONTEXT sources actually used. "
                 "Do not invent indices. If none were needed, write <CITATIONS>[]</CITATIONS>.")
            ]

    messages = [{"role": "system", "content": " ".join(rules)}]
    if evidence_mode == "web_live" and context_text and context_text.strip():
        messages[0]["content"] += (
            " Le CONTEXTE contient des resultats de recherche Web en direct, pas des documents locaux. "
            "Fonde la reponse uniquement sur ces sources Web et cite les indices effectivement utilises."
            if is_fr else
            " The CONTEXT contains live Web-search results, not local documents. "
            "Base the answer only on those Web sources and cite the indices actually used."
        )
    if evidence_mode in {"direct", "related"} and context_text and context_text.strip():
        if is_fr:
            messages[0]["content"] += (
                " Le contexte a été retenu comme preuve exploitable : produis obligatoirement une synthèse factuelle "
                "sourcée avant toute réserve."
            )
        else:
            messages[0]["content"] += (
                " The context was retained as usable evidence: always provide a sourced factual synthesis before any caveat."
            )
    if evidence_mode == "related" and context_text and context_text.strip():
        messages[0]["content"] += (
            " The context contains related evidence, but not direct evidence for every requested entity. "
            "State what is documented, name the documented case, explain why it is related, and explicitly say "
            "that it cannot establish the answer for the requested case with certainty. If related sources disagree, "
            "report the disagreement without selecting one as a conclusion for the requested case."
        )
    for m in (history or []):
        if isinstance(m, dict) and "role" in m and "content" in m:
            messages.append({"role": m["role"], "content": m["content"]})

    context_text = context_text or ""
    if not context_text.strip():
        user_content = question
        final_temperature = generation.temperature if temperature is None else temperature
        top_p = generation.top_p
    else:
        user_content = f"CONTEXTE:\n{context_text}\n\nQUESTION: {question}\nConsigne: réponds factuellement à partir du CONTEXTE uniquement."
        final_temperature = generation.strict_temperature if temperature is None else temperature
        top_p = generation.strict_top_p

    messages.append({"role": "user", "content": user_content})
    payload = {
        "model": model or generation.model,
        "messages": messages,
        "temperature": _adjust_temp(final_temperature),
        "top_p": top_p,
        "max_tokens": max_tokens if max_tokens is not None else generation.max_tokens,
        "stream": True  # STREAMING ACTIVÉ
    }
    
    yield from stream_from_payload(payload)
