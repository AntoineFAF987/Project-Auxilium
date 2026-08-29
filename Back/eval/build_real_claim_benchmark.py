from __future__ import annotations

"""Materialize the versioned benchmark from locally stored Auxilium answers.

The chat database is read-only. Contexts are replayed against the active v2 index
with the unchanged production retrieval parameters because historical final
contexts were not persisted in chat metadata.
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet

from api.index_singleton import idx
from rag_core.constants import FINAL_K, FUSE_ADJACENT_GAP, HYBRID_ALPHA, MAX_CONTEXT_CHARS, RETRIEVE_K, TOP_K_FAISS
from rag_core.faithfulness import extract_answer_claims
from rag_core.retrieval import clip_context_blocks, fuse_contiguous_passages
from runtime_settings import get_runtime_settings


BACK_DIR = Path(__file__).resolve().parents[1]
SELECTIONS = {
    "dev": [240, 178, 180, 162, 220, 90, 234],
    "test": [164, 202, 208, 218, 226, 124],
}

EXPECTED_STATUSES = {
    240: ["SUPPORTED"] * 4,
    178: ["SUPPORTED", "PARTIALLY_SUPPORTED"],
    180: ["SUPPORTED", "SUPPORTED", "SUPPORTED", "INFERRED"],
    162: ["PARTIALLY_SUPPORTED", "SUPPORTED", "INFERRED", "SUPPORTED", "INFERRED", "SUPPORTED"],
    220: ["SUPPORTED"] * 9,
    90: ["SUPPORTED", "SUPPORTED", "SUPPORTED", "PARTIALLY_SUPPORTED", "SUPPORTED", "SUPPORTED"],
    234: ["SUPPORTED", "INFERRED", "INFERRED"],
    164: ["SUPPORTED"] * 4,
    202: ["SUPPORTED"] * 5,
    208: ["SUPPORTED"] * 7,
    218: ["SUPPORTED"] * 5,
    226: ["SUPPORTED"] * 4,
    124: ["SUPPORTED"] * 5,
}


def _messages() -> dict[int, dict[str, Any]]:
    decrypt = Fernet((BACK_DIR / "api" / "chat_secret.key").read_bytes()).decrypt
    connection = sqlite3.connect(f"file:{BACK_DIR / 'api' / 'chats.db'}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT id, chat_id, role, content_enc, meta_json FROM chat_messages ORDER BY id"
    ).fetchall()
    result: dict[int, dict[str, Any]] = {}
    previous_by_chat: dict[str, sqlite3.Row] = {}
    for row in rows:
        previous = previous_by_chat.get(row["chat_id"])
        if row["role"] == "assistant" and previous and previous["role"] == "user":
            result[int(row["id"])] = {
                "question": decrypt(previous["content_enc"].encode()).decode("utf-8", errors="replace"),
                "answer": decrypt(row["content_enc"].encode()).decode("utf-8", errors="replace"),
                "metadata": json.loads(row["meta_json"] or "{}"),
            }
        previous_by_chat[row["chat_id"]] = row
    return result


def _context(question: str) -> list[dict[str, Any]]:
    settings = get_runtime_settings()
    retrieved, _ = idx.search(
        question,
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
    keys = (
        "text", "document_id", "chunk_uid", "fused_chunk_uids", "path", "page",
        "section", "heading_path", "block_ids", "document_metadata", "source_metadata",
    )
    return [{key: meta.get(key) for key in keys if meta.get(key) is not None} for _, meta in blocks]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    messages = _messages()
    splits = SELECTIONS if args.split == "all" else {args.split: SELECTIONS[args.split]}
    context_cache: dict[str, list[dict[str, Any]]] = {}
    materialized: dict[str, list[dict[str, Any]]] = {name: [] for name in splits}
    for split, ids in splits.items():
        for assistant_id in ids:
            row = messages[assistant_id]
            if row["question"] not in context_cache:
                context_cache[row["question"]] = _context(row["question"])
            context = context_cache[row["question"]]
            hidden = re.search(r"<CITATIONS>\s*\[?([\d,\s]*)\]?\s*</CITATIONS>", row["answer"], re.I)
            citations = [int(item) for item in re.findall(r"\d+", hidden.group(1))] if hidden else []
            claims = extract_answer_claims(row["answer"], cited_source_indices=citations)
            expected = EXPECTED_STATUSES[assistant_id]
            if len(claims) != len(expected):
                raise RuntimeError(
                    f"Annotation drift for chat-{assistant_id}: {len(claims)} claims, {len(expected)} labels"
                )
            expected_claims = []
            for claim, status in zip(claims, expected):
                absence = claim.claim_type == "EVIDENCE_ABSENCE"
                expected_claims.append({
                    "text": claim.verification_text or claim.text,
                    "status": status,
                    "expected_evidence_source_indices": [] if absence else [1, 2],
                    "expected_citation_status": (
                        "MATCHED" if claim.cited_source_indices else "NOT_APPLICABLE"
                    ),
                })
            item = {
                "schema_version": 2,
                "case_id": f"real-chat-{assistant_id}",
                "split": split,
                "origin": "real_auxilium_chat",
                "chat_assistant_id": assistant_id,
                "question": row["question"],
                "answer": row["answer"],
                "context_provenance": "replayed_active_v2_same_production_retrieval_parameters",
                "evidence": context,
                "cited_source_indices": citations,
                "expected_claims": expected_claims,
                "expected_statuses": expected,
                "expected_sources": [None if claim.claim_type == "EVIDENCE_ABSENCE" else [1, 2] for claim in claims],
                "expected_citation_statuses": [claim["expected_citation_status"] for claim in expected_claims],
                "expected_caveat": any(status != "SUPPORTED" for status in expected),
            }
            materialized[split].append(item)
            print(f"{split} chat-{assistant_id} context={len(context)} claims={len(claims)}")
            for index, claim in enumerate(claims, start=1):
                print(f"  {index}: [{claim.claim_type}] {claim.verification_text or claim.text}")
    if "dev" in materialized:
        source = materialized["dev"][0]
        first_evidence = source["evidence"][:1]
        mutations = [
            {
                "case_id": "controlled-real-3725-generalization",
                "question": "Peut-on tropicaliser un positionneur 3725 ?",
                "answer": "La tropicalisation complète du positionneur 3725 n'est plus possible.",
                "status": "UNSUPPORTED",
            },
            {
                "case_id": "controlled-real-3730-contradiction",
                "question": "Peut-on tropicaliser un positionneur 3730 ?",
                "answer": "La tropicalisation complète du positionneur 3730 est possible.",
                "status": "CONTRADICTED",
            },
        ]
        for mutation in mutations:
            target_split = "dev" if "generalization" in mutation["case_id"] else "test"
            if target_split not in materialized:
                continue
            materialized[target_split].append({
                "schema_version": 2,
                "case_id": mutation["case_id"],
                "split": target_split,
                "origin": "controlled_mutation_of_real_auxilium_case_240",
                "question": mutation["question"],
                "answer": mutation["answer"],
                "context_provenance": "real_chat_240_replayed_context",
                "evidence": first_evidence,
                "cited_source_indices": [],
                "expected_claims": [{
                    "text": mutation["answer"],
                    "status": mutation["status"],
                    "expected_evidence_source_indices": [1],
                    "expected_citation_status": "NOT_APPLICABLE",
                }],
                "expected_statuses": [mutation["status"]],
                "expected_sources": [1],
                "expected_citation_statuses": ["NOT_APPLICABLE"],
                "expected_caveat": True,
            })
    if args.write:
        output_dir = BACK_DIR / "eval" / "data"
        for split, items in materialized.items():
            output = output_dir / f"claim_faithfulness_real_{split}_v2.jsonl"
            output.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {output} ({len(items)} cases)")


if __name__ == "__main__":
    main()
