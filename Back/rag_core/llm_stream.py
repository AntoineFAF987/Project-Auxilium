# -*- coding: utf-8 -*-
"""
ask_mistral_with_context_stream — Version streaming pour affichage progressif
"""
from typing import List, Dict, Iterator
from .affect import detect_mood


def ask_mistral_with_context_stream(
    question: str,
    context_text: str,
    history: List[Dict] = None,
    model: str = None,
    temperature: float = 0.5,
    max_tokens: int = 1200,
    smalltalk_mode: bool = False,
    smalltalk_kind: str | None = None,
    roleplay_mode: bool = False,
    **kwargs,
) -> Iterator[str]:
    """
    Version streaming de ask_mistral_with_context.
    Yield des chunks de texte au fur et à mesure.
    """
    import os as _os, re as _re, requests as _requests

    api_key = _os.getenv("MISTRAL_API_KEY")
    if not api_key:
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
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model or _os.getenv("MISTRAL_MODEL","mistral-small-latest"),
            "messages": messages,
            "temperature": _adjust_temp(max(temperature,0.9)),
            "top_p": 0.9,
            "max_tokens": min(max_tokens,220),
            "stream": True  # STREAMING ACTIVÉ
        }
        with _requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=60, stream=True) as r:
            if r.status_code == 401: raise RuntimeError("401 Unauthorized: vérifie MISTRAL_API_KEY.")
            if r.status_code == 429: raise RuntimeError("429 Too Many Requests: quota/ratelimit.")
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]
                if line.strip() == '[DONE]':
                    break
                try:
                    import json
                    chunk = json.loads(line)
                    delta = chunk.get('choices', [{}])[0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        yield content
                except:
                    continue
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
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": _os.getenv("MISTRAL_MODEL","mistral-small-latest"),
            "messages": messages,
            "temperature": _adjust_temp(max(temperature,0.65)),
            "top_p": 0.9,
            "max_tokens": min(max_tokens,200),
            "stream": True  # STREAMING ACTIVÉ
        }
        with _requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=60, stream=True) as r:
            if r.status_code == 401: raise RuntimeError("401 Unauthorized: vérifie MISTRAL_API_KEY.")
            if r.status_code == 429: raise RuntimeError("429 Too Many Requests: quota/ratelimit.")
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]
                if line.strip() == '[DONE]':
                    break
                try:
                    import json
                    chunk = json.loads(line)
                    delta = chunk.get('choices', [{}])[0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        yield content
                except:
                    continue
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
            rules.append("Si la question demande des faits précis sans source fournie, dis-le plutôt que de deviner.")
        else:
            rules += [
                'Utilise UNIQUEMENT le CONTEXTE fourni ; si l\'info manque, réponds exactement : "Je ne sais pas".',
                "N'AJOUTE JAMAIS d'informations, comparaisons ou exemples qui ne sont pas explicitement dans le CONTEXTE.",
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
            rules.append("If precise facts are requested and no source is provided, say so rather than guessing.")
        else:
            rules += [
                "Answer ONLY from the provided CONTEXT; if missing, reply exactly: \"I don't know\".",
                "NEVER add information, comparisons, or examples that are not explicitly in the CONTEXT.",
                "Even in strict mode, provide a complete and structured answer based on context.",
                ("At the very end of your answer, add a line <CITATIONS>[i1,i2,...]</CITATIONS> "
                 "where i1,i2,... are the [1..N] indices of CONTEXT sources actually used. "
                 "Do not invent indices. If none were needed, write <CITATIONS>[]</CITATIONS>.")
            ]

    messages = [{"role": "system", "content": " ".join(rules)}]
    for m in (history or []):
        if isinstance(m, dict) and "role" in m and "content" in m:
            messages.append({"role": m["role"], "content": m["content"]})

    context_text = context_text or ""
    if not context_text.strip():
        user_content = question; final_temperature, top_p = 0.6, 0.9
    else:
        user_content = f"CONTEXTE:\n{context_text}\n\nQUESTION: {question}\nConsigne: réponds factuellement à partir du CONTEXTE uniquement."
        final_temperature, top_p = 0.45, 1.0

    messages.append({"role": "user", "content": user_content})
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": _os.getenv("MISTRAL_MODEL","mistral-small-latest"),
        "messages": messages,
        "temperature": _adjust_temp(final_temperature),
        "top_p": top_p,
        "max_tokens": 1200,
        "stream": True  # STREAMING ACTIVÉ
    }
    
    with _requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=60, stream=True) as r:
        if r.status_code == 401: raise RuntimeError("401 Unauthorized: vérifie MISTRAL_API_KEY.")
        if r.status_code == 429: raise RuntimeError("429 Too Many Requests: quota/ratelimit.")
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode('utf-8')
            if line.startswith('data: '):
                line = line[6:]
            if line.strip() == '[DONE]':
                break
            try:
                import json
                chunk = json.loads(line)
                delta = chunk.get('choices', [{}])[0].get('delta', {})
                content = delta.get('content', '')
                if content:
                    yield content
            except:
                continue
