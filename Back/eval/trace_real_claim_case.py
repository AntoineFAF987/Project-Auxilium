from __future__ import annotations

import json
from typing import Any, Dict, List

from api.index_singleton import idx
from rag_core.constants import FINAL_K, FUSE_ADJACENT_GAP, HYBRID_ALPHA, MAX_CONTEXT_CHARS, RETRIEVE_K, TOP_K_FAISS
from rag_core.faithfulness import (
    _candidate_evidence,
    _normalize,
    _numbers,
    _product_ids,
    _run_nli_batch,
    _tokens,
    extract_answer_claims,
    verify_answer_claims,
)
from rag_core.retrieval import clip_context_blocks, fuse_contiguous_passages
from runtime_settings import get_runtime_settings


QUESTION = "J’ai reçu une question d’un commercial. Il me demande s’il est possible de tropicaliser un positionneur 3725 ?"
ANSWER = (
    "D’après le contexte fourni, la tropicalisation du TROVIS 3730 n’est plus possible en raison d’un changement "
    "de technologie au niveau du détecteur de position. Cette information est explicitement mentionnée dans le "
    "document [1], qui précise que l’application d’un vernis supplémentaire modifierait le comportement du détecteur.\n\n"
    "Cependant, aucune information n’est donnée dans le contexte concernant le positionneur 3725. Le document [1] "
    "traite uniquement du 3730, et les autres sources ne mentionnent ni ce modèle ni sa tropicalisation."
)


def run_trace() -> Dict[str, Any]:
    settings = get_runtime_settings()
    retrieved, _ = idx.search(
        QUESTION,
        retrieve_k=RETRIEVE_K,
        top_k_faiss=TOP_K_FAISS,
        hybrid_alpha=HYBRID_ALPHA,
        use_rerank=settings.features.enable_reranker,
        allowed_sources={"email", "pdf", "file"},
    )
    blocks = clip_context_blocks(
        fuse_contiguous_passages(retrieved, gap=FUSE_ADJACENT_GAP),
        max_chars=MAX_CONTEXT_CHARS,
        keep=FINAL_K,
    )
    evidence_blocks = [dict(block[1]) for block in blocks]
    claims = extract_answer_claims(ANSWER, cited_source_indices=[1])
    evidence_by_claim = []
    for claim in claims:
        evidence, _ = _candidate_evidence(claim, evidence_blocks)
        evidence_by_claim.append(evidence)
    nli, _, model_name = _run_nli_batch(claims, evidence_by_claim, use_nli=True)
    review = verify_answer_claims(
        ANSWER,
        evidence_blocks,
        cited_source_indices=[1],
        use_nli=True,
    )
    trace_claims: List[Dict[str, Any]] = []
    for claim, candidates, nli_result, verification in zip(claims, evidence_by_claim, nli, review.claims):
        trace_claims.append({
            "claim_id": claim.claim_id,
            "claim": claim.text,
            "verification_text": claim.verification_text,
            "normalized": _normalize(claim.verification_text or claim.text),
            "tokens": sorted(_tokens(claim.verification_text or claim.text)),
            "product_ids": sorted(_product_ids(claim.verification_text or claim.text)),
            "numbers": sorted(_numbers(claim.verification_text or claim.text)),
            "cited_source_indices": list(claim.cited_source_indices),
            "candidate_evidence": [item.to_dict() for item in candidates[:3]],
            "nli_label": nli_result.label,
            "nli_score": nli_result.score,
            "nli_entailment_score": nli_result.entailment_score,
            "nli_contradiction_score": nli_result.contradiction_score,
            "citation_correct": verification.citation_correct,
            "citation_status": verification.citation_status.value,
            "citation_reason": verification.citation_reason,
            "final_status": verification.status.value,
            "final_reason": verification.reason,
        })
    return {
        "question": QUESTION,
        "answer": ANSWER,
        "model": model_name,
        "final_blocks": [
            {
                "source_index": index,
                "file": meta.get("file"),
                "document_id": meta.get("document_id"),
                "chunk_uid": meta.get("chunk_uid"),
                "fused_chunk_uids": meta.get("fused_chunk_uids"),
                "text": meta.get("text"),
            }
            for index, (_, meta) in enumerate(blocks, start=1)
        ],
        "claims": trace_claims,
        "review": review.to_dict(),
    }


if __name__ == "__main__":
    print(json.dumps(run_trace(), ensure_ascii=False, indent=2))
