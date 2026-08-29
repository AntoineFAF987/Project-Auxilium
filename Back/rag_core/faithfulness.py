from __future__ import annotations

"""Non-destructive, post-generation claim-level faithfulness review."""

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import re
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import unicodedata

_NLI_MODEL = None
_NLI_AVAILABLE = None
NLI_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
NLI_MODEL_FALLBACK = "cross-encoder/nli-deberta-v3-small"

SUPPORTED_THRESHOLD = 0.72
PARTIAL_THRESHOLD = 0.45
INFERENCE_THRESHOLD = 0.24
NLI_ENTAILMENT_THRESHOLD = 0.65
NLI_CONTRADICTION_THRESHOLD = 0.60

_STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "cette", "dans", "de", "des",
    "du", "elle", "en", "est", "et", "il", "la", "le", "les", "leur",
    "leurs", "mais", "ou", "par", "pas", "plus", "pour", "que", "qui",
    "son", "sur", "un", "une", "the", "an", "and", "are", "as", "at", "by", "for",
    "from", "in", "is", "it", "of", "on", "or", "that", "to", "was",
    "were", "with",
}
_NEGATIONS = {
    "aucun", "aucune", "impossible", "interdit", "jamais", "non", "not",
    "never", "no", "cannot", "can't", "without", "sans",
}
_RECOMMENDATION_PATTERNS = (
    r"^(?:je |nous )?(?:recommande|conseille|suggere)",
    r"^(?:vous |tu )?(?:devriez|pourriez)",
    r"^il (?:faut|faudrait|est recommande de|serait preferable de)",
    r"^(?:contactez|verifiez|veillez a|consultez)\b",
    r"^(?:we |i )?(?:recommend|suggest|advise)",
    r"^(?:you should|you could|it is recommended)",
)
_NON_FACTUAL_PATTERNS = (
    r"^(?:bonjour|bonsoir|merci|avec plaisir|en resume|en conclusion)\b",
    r"^(?:voici|here is|in summary|to conclude)\b[^.!?]*:?$",
    r"^(?:the key points|here are the key points|activites professionnelles|coordonnees|poste et role)\b[^.!?]*:?$",
    r"^(?:i.ve included|let me know)\b",
)


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    INFERRED = "INFERRED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


class CitationStatus(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    MATCHED = "MATCHED"
    MISMATCHED = "MISMATCHED"


@dataclass(frozen=True)
class AnswerClaim:
    claim_id: str
    text: str
    start: int
    end: int
    claim_type: str = "FACTUAL"
    cited_source_indices: Tuple[int, ...] = ()
    verification_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["cited_source_indices"] = list(self.cited_source_indices)
        return data


@dataclass(frozen=True)
class ClaimEvidence:
    evidence_id: str
    text: str
    start: int
    end: int
    support_score: float
    source_index: int
    document_id: Optional[str] = None
    chunk_uid: Optional[str] = None
    chunk_uids: Tuple[str, ...] = ()
    path: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    heading_path: Tuple[str, ...] = ()
    block_ids: Tuple[str, ...] = ()
    email_thread_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["heading_path"] = list(self.heading_path)
        data["block_ids"] = list(self.block_ids)
        data["chunk_uids"] = list(self.chunk_uids)
        return data


@dataclass(frozen=True)
class ClaimVerification:
    claim: AnswerClaim
    status: ClaimStatus
    confidence: float
    evidence: Tuple[ClaimEvidence, ...] = ()
    reason: str = ""
    citation_correct: Optional[bool] = None
    citation_status: CitationStatus = CitationStatus.NOT_APPLICABLE
    citation_reason: Optional[str] = None
    nli_label: Optional[str] = None
    nli_score: Optional[float] = None
    nli_entailment_score: Optional[float] = None
    nli_contradiction_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "status": self.status.value,
            "confidence": self.confidence,
            "evidence": [item.to_dict() for item in self.evidence],
            "reason": self.reason,
            "citation_correct": self.citation_correct,
            "citation_status": self.citation_status.value,
            "citation_reason": self.citation_reason,
            "nli_label": self.nli_label,
            "nli_score": self.nli_score,
            "nli_entailment_score": self.nli_entailment_score,
            "nli_contradiction_score": self.nli_contradiction_score,
        }


@dataclass(frozen=True)
class FaithfulnessReview:
    status: str
    claims: Tuple[ClaimVerification, ...]
    extraction_ms: float
    verification_ms: float
    total_ms: float
    evidence_chunks_inspected: int
    model_calls: int
    model_name: Optional[str]
    caveat_required: bool

    def to_dict(self) -> Dict[str, Any]:
        counts = {status.value: 0 for status in ClaimStatus}
        for item in self.claims:
            counts[item.status.value] += 1
        return {
            "status": self.status,
            "claims": [item.to_dict() for item in self.claims],
            "claim_count": len(self.claims),
            "status_counts": counts,
            "extraction_ms": self.extraction_ms,
            "verification_ms": self.verification_ms,
            "total_ms": self.total_ms,
            "evidence_chunks_inspected": self.evidence_chunks_inspected,
            "model_calls": self.model_calls,
            "model_name": self.model_name,
            "caveat_required": self.caveat_required,
        }

    def legacy_summary(self) -> Dict[str, Any]:
        problematic = [item for item in self.claims if item.status != ClaimStatus.SUPPORTED]
        worst = next((item for item in problematic if item.status == ClaimStatus.CONTRADICTED), None)
        worst = worst or next((item for item in problematic if item.status == ClaimStatus.UNSUPPORTED), None)
        worst = worst or (problematic[0] if problematic else None)
        return {
            "faithful": not problematic,
            "score": min((item.confidence for item in self.claims), default=1.0),
            "label": worst.status.value.lower() if worst else "entailment",
            "reason": worst.reason if worst else "All factual claims are supported",
        }


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", (value or "").casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return normalized.replace("’", "'").replace("‘", "'")


def _tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+(?:[_.-][a-z0-9]+)*", _normalize(value))
        if len(token) > 1 and token not in _STOPWORDS
    }


def _product_ids(value: str) -> set[str]:
    return {
        token for token in re.findall(r"\b\d{3,6}\b", value or "")
        if not (1900 <= int(token) <= 2099)
    }


def _numbers(value: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:[.,]\d+)?\b", value or ""))


def _polarity(value: str) -> bool:
    normalized = f" {_normalize(value)} "
    explicit = any(re.search(rf"\b{re.escape(marker)}\b", normalized) for marker in _NEGATIONS)
    grammatical = bool(
        re.search(r"\bn[' ]?(?:est|a|ont|sont)\b[^.!?]{0,30}\b(?:pas|plus)\b", normalized)
        or " no longer " in normalized
    )
    return explicit or grammatical


def _clean_answer(answer: str) -> str:
    cleaned = re.sub(r"<CITATIONS>.*?</CITATIONS>", "", answer or "", flags=re.I | re.S)
    cleaned = re.sub(r"\[WEB\].*?\[/WEB\]", "", cleaned, flags=re.I | re.S)
    return cleaned.strip()


def _verification_text(value: str) -> str:
    """Remove presentation/attribution syntax without changing displayed text."""
    text = re.sub(r"(?:\*\*|__|`)", "", value or "")
    text = re.sub(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]", " ", text)
    text = re.sub(
        r"^\s*(?:d['’]apr[eè]s|selon)\s+(?:le\s+)?contexte(?:\s+fourni)?\s*[,,:-]\s*",
        "", text, flags=re.I,
    )
    text = re.sub(r"^\s*(?:cependant|toutefois|en revanche)\s*[,,:-]\s*", "", text, flags=re.I)
    attributed = re.match(
        r"^.*?\bqui\s+(?:pr[eé]cise|indique|mentionne|confirme)\s+que\s+(.+)$",
        text, flags=re.I | re.S,
    )
    if attributed:
        text = attributed.group(1)
    return re.sub(r"\s+", " ", text).strip()


_SENTENCE_PATTERN = re.compile(r".+?(?:\.(?=\s|$)|[!?](?=\s|$)|\n|$)", re.S)


def _claim_units(sentence: str) -> List[Tuple[str, int, int]]:
    """Return explainable clause units, including explicit causal clauses."""
    units: List[Tuple[str, int, int]] = []
    for segment_match in re.finditer(r"[^;]+", sentence):
        raw = segment_match.group(0)
        stripped = raw.strip()
        if not stripped:
            continue
        local_start = segment_match.start() + len(raw) - len(raw.lstrip())
        causal = re.search(r"\s+(?:a cause de|because of)\s+", _normalize(stripped))
        if not causal:
            units.append((stripped, local_start, local_start + len(stripped)))
            continue
        before = stripped[:causal.start()].rstrip(" ,")
        cause = stripped[causal.end():].strip(" ,.!?")
        if before:
            units.append((before + ".", local_start, local_start + causal.start()))
        if cause:
            raw_cause_start = local_start + causal.end()
            units.append((f"La cause est {cause}.", raw_cause_start, raw_cause_start + len(cause)))
    return units


def extract_answer_claims(
    answer: str, *, cited_source_indices: Sequence[int] = ()
) -> List[AnswerClaim]:
    """Extract factual sentences/clauses with stable response-local IDs."""
    cleaned = _clean_answer(answer)
    answer_has_inline_citations = bool(re.search(r"\[\d+\]", cleaned))
    claims: List[AnswerClaim] = []
    for sentence_match in _SENTENCE_PATTERN.finditer(cleaned):
        raw_sentence = sentence_match.group(0)
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        sentence_start = sentence_match.start() + len(raw_sentence) - len(raw_sentence.lstrip())
        for unit_text, unit_start, unit_end in _claim_units(sentence):
            text = unit_text.strip(" \t\r\n-*•")
            check_text = _verification_text(text)
            normalized = _normalize(check_text)
            if len(_tokens(check_text)) < 2:
                continue
            if check_text.rstrip().endswith(":") or normalized.endswith(" max."):
                continue
            if re.search(r"\b(?:je ne peux pas repondre|i cannot answer|i can.t answer)\b", normalized):
                continue
            if any(re.search(pattern, normalized) for pattern in _RECOMMENDATION_PATTERNS):
                continue
            if any(re.search(pattern, normalized) for pattern in _NON_FACTUAL_PATTERNS):
                continue
            start = sentence_start + unit_start
            end = sentence_start + unit_end
            absence = bool(re.search(
                r"\b(?:aucune?\s+(?:information|mention)|ne\s+(?:mentionne|mentionnent|contient|contiennent)|seul document.*\bnon\b)\b",
                normalized,
            ))
            claim_type = (
                "EVIDENCE_ABSENCE" if absence else
                "QUANTITATIVE" if _numbers(check_text) else
                "NEGATED_FACT" if _polarity(check_text) else "FACTUAL"
            )
            local_citations = tuple(int(item) for item in re.findall(r"\[(\d+)\]", text))
            citations = local_citations
            if not citations and not answer_has_inline_citations:
                citations = tuple(dict.fromkeys(int(i) for i in cited_source_indices if int(i) > 0))
            digest = hashlib.sha1(f"{start}:{end}:{text}".encode("utf-8")).hexdigest()[:10]
            claims.append(AnswerClaim(
                claim_id=f"claim-{len(claims) + 1:03d}-{digest}",
                text=text,
                start=start,
                end=end,
                claim_type=claim_type,
                cited_source_indices=tuple(dict.fromkeys(citations)),
                verification_text=check_text,
            ))
    return claims


def _sentences_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    spans = []
    for match in _SENTENCE_PATTERN.finditer(text or ""):
        sentence = match.group(0).strip()
        if sentence:
            leading = len(match.group(0)) - len(match.group(0).lstrip())
            start = match.start() + leading
            spans.append((sentence, start, start + len(sentence)))
    if not spans and text.strip():
        start = len(text) - len(text.lstrip())
        spans.append((text.strip(), start, start + len(text.strip())))
    return spans


def _passages_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Sentence spans plus adjacent pairs for claims supported across a boundary."""
    sentences = _sentences_with_offsets(text)
    passages = list(sentences)
    for first, second in zip(sentences, sentences[1:]):
        passages.append((text[first[1]:second[2]].strip(), first[1], second[2]))
    stripped = text.strip()
    if stripped and len(sentences) > 1:
        start = len(text) - len(text.lstrip())
        passages.append((stripped, start, start + len(stripped)))
    return passages


def _evidence_score(claim: str, passage: str) -> float:
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return 0.0
    overlap = len(claim_tokens & _tokens(passage)) / len(claim_tokens)
    claim_products = _product_ids(claim)
    if claim_products and not claim_products.issubset(_product_ids(passage)):
        overlap *= 0.2
    claim_numbers = _numbers(claim) - claim_products
    if claim_numbers and not claim_numbers.issubset(_numbers(passage)):
        overlap *= 0.45
    return overlap


def _email_thread_id(meta: Mapping[str, Any]) -> Optional[str]:
    document_meta = meta.get("document_metadata") or {}
    return document_meta.get("thread_id") or (meta.get("source_metadata") or {}).get("thread_id")


def _candidate_evidence(
    claim: AnswerClaim, evidence_blocks: Sequence[Mapping[str, Any]]
) -> Tuple[List[ClaimEvidence], int]:
    candidates: List[ClaimEvidence] = []
    for source_index, meta in enumerate(evidence_blocks, start=1):
        text = str(meta.get("text") or "")
        block_candidates: List[ClaimEvidence] = []
        check_text = claim.verification_text or claim.text
        for passage, start, end in _passages_with_offsets(text):
            score = _evidence_score(check_text, passage)
            item = ClaimEvidence(
                evidence_id=f"evidence-{source_index}-{start}-{end}", text=passage,
                start=start, end=end, support_score=score, source_index=source_index,
                document_id=meta.get("document_id"), chunk_uid=meta.get("chunk_uid"),
                chunk_uids=tuple(
                    meta.get("fused_chunk_uids")
                    or ([meta.get("chunk_uid")] if meta.get("chunk_uid") else [])
                ),
                path=meta.get("path"), page=meta.get("page"), section=meta.get("section"),
                heading_path=tuple(meta.get("heading_path") or ()),
                block_ids=tuple(meta.get("block_ids") or ()), email_thread_id=_email_thread_id(meta),
            )
            block_candidates.append(item)
        block_candidates.sort(
            key=lambda item: (item.support_score, len(item.text) if item.support_score < INFERENCE_THRESHOLD else 0),
            reverse=True,
        )
        candidates.extend(block_candidates[:3])
    candidates.sort(key=lambda item: item.support_score, reverse=True)
    return candidates, len(evidence_blocks)


def _load_nli_model():
    global _NLI_MODEL, _NLI_AVAILABLE
    if _NLI_AVAILABLE is not None:
        return _NLI_MODEL
    try:
        from transformers import pipeline
        _NLI_MODEL = pipeline("text-classification", model=NLI_MODEL_NAME, device=-1)
        _NLI_AVAILABLE = True
        return _NLI_MODEL
    except Exception:
        try:
            from transformers import pipeline
            _NLI_MODEL = pipeline("text-classification", model=NLI_MODEL_FALLBACK, device=-1)
            _NLI_AVAILABLE = True
            return _NLI_MODEL
        except Exception:
            _NLI_AVAILABLE = False
            return None


@dataclass(frozen=True)
class _NLIResult:
    label: Optional[str] = None
    score: Optional[float] = None
    entailment_score: Optional[float] = None
    contradiction_score: Optional[float] = None


def _run_nli_batch(
    claims: Sequence[AnswerClaim], evidences: Sequence[Sequence[ClaimEvidence]],
    *, use_nli: bool, nli_model: Any = None,
) -> Tuple[List[_NLIResult], int, Optional[str]]:
    if not use_nli or not claims:
        return [_NLIResult() for _ in claims], 0, None
    model = nli_model if nli_model is not None else _load_nli_model()
    if model is None:
        return [_NLIResult() for _ in claims], 0, None
    inputs = [
        f"{' '.join(item.text for item in evidence[:2])} [SEP] {claim.verification_text or claim.text}"
        for claim, evidence in zip(claims, evidences)
    ]
    try:
        raw_results = model(inputs, truncation=True, batch_size=min(8, len(inputs)), top_k=None)
        if isinstance(raw_results, dict):
            raw_results = [raw_results]
        parsed: List[_NLIResult] = []
        for raw in raw_results:
            alternatives = raw if isinstance(raw, list) else [raw]
            best = max(alternatives, key=lambda item: float(item.get("score", 0.0)))
            scores = {
                str(item.get("label") or "").upper(): float(item.get("score") or 0.0)
                for item in alternatives
            }
            parsed.append(_NLIResult(
                label=str(best.get("label") or "").upper(),
                score=float(best.get("score") or 0.0),
                entailment_score=max((v for k, v in scores.items() if "ENTAIL" in k), default=None),
                contradiction_score=max((v for k, v in scores.items() if "CONTRADICTION" in k), default=None),
            ))
        while len(parsed) < len(claims):
            parsed.append(_NLIResult())
        model_name = getattr(getattr(model, "model", None), "name_or_path", None) or NLI_MODEL_NAME
        return parsed[:len(claims)], 1, str(model_name)
    except Exception:
        return [_NLIResult() for _ in claims], 0, None


def _deterministic_contradiction(claim: str, passage: str, score: float) -> bool:
    return (
        score >= SUPPORTED_THRESHOLD
        and not passage.rstrip().endswith("?")
        and _polarity(claim) != _polarity(passage)
    )


def _looks_multi_fact(value: str) -> bool:
    normalized = f" {_normalize(value)} "
    return any(connector in normalized for connector in (
        " et ", " ainsi que ", " mais ", " and ", " as well as ", " but ",
    ))


def _absence_supported(
    claim: AnswerClaim, evidence_blocks: Sequence[Mapping[str, Any]]
) -> Optional[bool]:
    if claim.claim_type != "EVIDENCE_ABSENCE":
        return None
    text = claim.verification_text or claim.text
    products = _product_ids(text)
    cited = set(claim.cited_source_indices)
    normalized = _normalize(text)
    excluded_product = re.search(r"\bnon\s+(\d{3,6})\b", normalized)
    if excluded_product:
        products = {excluded_product.group(1)}
    if "autres sources" in normalized or "autres documents" in normalized:
        cited_documents = {
            block.get("document_id")
            for index, block in enumerate(evidence_blocks, start=1)
            if index in cited and block.get("document_id")
        }
        inspected = [
            str(block.get("text") or "")
            for index, block in enumerate(evidence_blocks, start=1)
            if index not in cited and block.get("document_id") not in cited_documents
        ]
    else:
        inspected = [str(block.get("text") or "") for block in evidence_blocks]
    if products:
        present = set().union(*(_product_ids(item) for item in inspected)) if inspected else set()
        return products.isdisjoint(present)
    target_match = re.search(
        r"aucune mention (?:d[' ]?|de |du |des )?(.+?) n[' ]?(?:est|apparait)", normalized
    )
    if not target_match:
        return None
    target = re.sub(r"^(?:un|une|le|la|les)\s+", "", target_match.group(1)).strip()
    if len(_tokens(target)) < 1:
        return None
    normalized_context = " ".join(_normalize(item) for item in inspected)
    return target not in normalized_context


def _citation_assessment(
    claim: AnswerClaim, evidence: Sequence[ClaimEvidence]
) -> Tuple[Optional[bool], CitationStatus, Optional[str]]:
    cited = set(claim.cited_source_indices)
    if not cited:
        return None, CitationStatus.NOT_APPLICABLE, None
    supporting = [item for item in evidence if item.support_score >= PARTIAL_THRESHOLD]
    if any(item.source_index in cited for item in supporting):
        return True, CitationStatus.MATCHED, "The cited context block contains supporting evidence"
    cited_documents = {
        item.document_id for item in evidence
        if item.source_index in cited and item.document_id
    }
    if cited_documents and any(item.document_id in cited_documents for item in supporting):
        return True, CitationStatus.MATCHED, "Supporting evidence is in another chunk of the cited document"
    cited_threads = {
        item.email_thread_id for item in evidence
        if item.source_index in cited and item.email_thread_id
    }
    if cited_threads and any(item.email_thread_id in cited_threads for item in supporting):
        return True, CitationStatus.MATCHED, "Supporting evidence is in another message of the cited email thread"
    return False, CitationStatus.MISMATCHED, "No supporting span was found in the cited source or document"


def verify_answer_claims(
    answer: str, evidence_blocks: Sequence[Mapping[str, Any]], *,
    cited_source_indices: Sequence[int] = (), use_nli: bool = True, nli_model: Any = None,
) -> FaithfulnessReview:
    total_started = perf_counter()
    extraction_started = perf_counter()
    claims = extract_answer_claims(answer, cited_source_indices=cited_source_indices)
    extraction_ms = (perf_counter() - extraction_started) * 1000.0
    verification_started = perf_counter()
    evidence_by_claim: List[List[ClaimEvidence]] = []
    inspected = 0
    for claim in claims:
        evidence, count = _candidate_evidence(claim, evidence_blocks)
        evidence_by_claim.append(evidence)
        inspected += count
    nli_results = [_NLIResult() for _ in claims]
    nli_indices = []
    for index, (claim, evidence) in enumerate(zip(claims, evidence_by_claim)):
        best = evidence[0] if evidence else None
        lexical = best.support_score if best else 0.0
        check_text = claim.verification_text or claim.text
        deterministic = bool(best and _deterministic_contradiction(check_text, best.text, lexical))
        if claim.claim_type != "EVIDENCE_ABSENCE" and lexical < SUPPORTED_THRESHOLD and not deterministic:
            nli_indices.append(index)
    selected_nli, model_calls, model_name = _run_nli_batch(
        [claims[index] for index in nli_indices],
        [evidence_by_claim[index] for index in nli_indices],
        use_nli=use_nli,
        nli_model=nli_model,
    )
    for index, result in zip(nli_indices, selected_nli):
        nli_results[index] = result

    verifications: List[ClaimVerification] = []
    for claim, evidence, nli in zip(claims, evidence_by_claim, nli_results):
        declarative = [item for item in evidence if not item.text.rstrip().endswith("?")]
        best = declarative[0] if declarative else (evidence[0] if evidence else None)
        lexical = best.support_score if best else 0.0
        citation_correct, citation_status, citation_reason = _citation_assessment(claim, evidence)
        check_text = claim.verification_text or claim.text
        absence_supported = _absence_supported(claim, evidence_blocks)
        if absence_supported is True and claim.cited_source_indices:
            citation_correct = True
            citation_status = CitationStatus.MATCHED
            citation_reason = "The cited document establishes the scope; absence was checked across the remaining context"
        aligned_evidence = any(
            item.support_score >= PARTIAL_THRESHOLD
            and not item.text.rstrip().endswith("?")
            and _polarity(check_text) == _polarity(item.text)
            for item in evidence
        )
        deterministic_contradiction = bool(
            best and not aligned_evidence
            and _deterministic_contradiction(check_text, best.text, lexical)
        )
        contradiction = deterministic_contradiction
        if (
            (nli.contradiction_score or 0.0) >= NLI_CONTRADICTION_THRESHOLD
            and lexical >= PARTIAL_THRESHOLD
            and not aligned_evidence
            and not (best and best.text.rstrip().endswith("?"))
            and _polarity(check_text) != _polarity(best.text if best else "")
        ):
            contradiction = True
        if absence_supported is True:
            status, confidence, reason = ClaimStatus.SUPPORTED, 1.0, "The stated absence is confirmed across the inspected context"
        elif absence_supported is False:
            status, confidence, reason = ClaimStatus.CONTRADICTED, 1.0, "The context contains the information claimed to be absent"
        elif contradiction:
            status, confidence, reason = ClaimStatus.CONTRADICTED, max(lexical, nli.contradiction_score or 0.0), "The closest evidence expresses the opposite polarity"
        elif lexical >= SUPPORTED_THRESHOLD:
            status, confidence, reason = ClaimStatus.SUPPORTED, lexical, "Direct lexical support in the selected evidence"
        elif (nli.entailment_score or 0.0) >= NLI_ENTAILMENT_THRESHOLD:
            status, confidence, reason = ClaimStatus.SUPPORTED, max(lexical, nli.entailment_score or 0.0), "Entailed by the selected evidence"
        elif lexical >= PARTIAL_THRESHOLD or (
            lexical >= INFERENCE_THRESHOLD and _looks_multi_fact(claim.text)
        ):
            status, confidence, reason = ClaimStatus.PARTIALLY_SUPPORTED, lexical, "Only part of the factual content is explicit in the evidence"
        elif lexical >= INFERENCE_THRESHOLD and (
            not _product_ids(claim.text) or _product_ids(claim.text).issubset(_product_ids(best.text if best else ""))
        ):
            status, confidence, reason = ClaimStatus.INFERRED, lexical, "Related evidence exists, but the claim is not stated explicitly"
        else:
            status, confidence, reason = ClaimStatus.UNSUPPORTED, 1.0 - lexical, "No selected evidence directly supports this claim"
        if citation_correct is False and status == ClaimStatus.SUPPORTED:
            cited = set(claim.cited_source_indices)
            cited_documents = {
                item.document_id for item in evidence
                if item.source_index in cited and item.document_id
            }
            if (
                best and (
                    best.source_index in cited
                    or (best.document_id and best.document_id in cited_documents)
                )
                and (nli.entailment_score or 0.0) >= NLI_ENTAILMENT_THRESHOLD
            ):
                citation_correct = True
                citation_status = CitationStatus.MATCHED
                citation_reason = "The cited source contains the premise accepted by NLI"
            else:
                citation_reason = "The fact is supported, but the cited source does not contain its evidence"
        associated = tuple(item for item in evidence if item.support_score >= INFERENCE_THRESHOLD)
        verifications.append(ClaimVerification(
            claim=claim, status=status, confidence=min(1.0, max(0.0, confidence)),
            evidence=associated, reason=reason, citation_correct=citation_correct,
            citation_status=citation_status, citation_reason=citation_reason,
            nli_label=nli.label, nli_score=nli.score,
            nli_entailment_score=nli.entailment_score,
            nli_contradiction_score=nli.contradiction_score,
        ))
    verification_ms = (perf_counter() - verification_started) * 1000.0
    caveat_required = any(
        item.status != ClaimStatus.SUPPORTED
        or (item.status == ClaimStatus.SUPPORTED and item.citation_status == CitationStatus.MISMATCHED)
        for item in verifications
    )
    return FaithfulnessReview(
        status="CAVEAT" if caveat_required else "OK", claims=tuple(verifications),
        extraction_ms=extraction_ms, verification_ms=verification_ms,
        total_ms=(perf_counter() - total_started) * 1000.0,
        evidence_chunks_inspected=inspected, model_calls=model_calls,
        model_name=model_name, caveat_required=caveat_required,
    )


def check_faithfulness(answer: str, context: str, threshold: float = 0.5) -> Dict[str, Any]:
    """Backward-compatible aggregate backed by claim-level verification."""
    if not context or not context.strip():
        return {"faithful": True, "score": 1.0, "label": "no_context", "reason": "Pas de contexte strict"}
    if not answer or not answer.strip():
        return {"faithful": False, "score": 0.0, "label": "empty_answer", "reason": "Réponse vide"}
    return verify_answer_claims(answer, [{"text": context}], use_nli=True).legacy_summary()


def should_fallback_to_general(faithfulness_check: Dict[str, Any], strict_mode: bool) -> bool:
    if not strict_mode or faithfulness_check.get("label") in {"no_context", "nli_unavailable", "error"}:
        return False
    return not bool(faithfulness_check.get("faithful")) and float(faithfulness_check.get("score", 1.0)) < 0.4
