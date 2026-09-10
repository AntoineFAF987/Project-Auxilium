"""Enrich a completed production retrieval report with local final-context text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gold import load_gold_cases

BACK_DIR = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    gold = {case.id: case for case in load_gold_cases(BACK_DIR / "eval" / "data" / "gold_v1.jsonl")}
    targets = {str(uid) for row in report["results"] for uid in row.get("context_chunk_uids", []) if uid}
    manifest = json.loads((BACK_DIR / "data" / "index_rag" / "manifest.json").read_text(encoding="utf-8"))
    corpus = BACK_DIR / "data" / "index_rag" / manifest["artifacts"]["corpus"]
    by_uid = {}
    with corpus.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if str(row.get("chunk_uid")) in targets:
                by_uid[str(row["chunk_uid"])] = row
    failures = []
    for row in report["results"]:
        if not row.get("scored") or (row.get("document_hit") and row.get("context_recall") == 1.0):
            continue
        uids = row.get("context_chunk_uids", [])
        failures.append({
            "id": row["id"], "question": row["question"], "expected_document_ids": gold[row["id"]].expected_document_ids,
            "document_hit": row.get("document_hit"), "recall_at_10": row.get("recall@10"),
            "top_documents": row.get("retrieved_documents", []), "context_documents": row.get("context_documents", []),
            "required_facts_missing": row.get("required_facts_missing", []), "sufficiency": row.get("sufficiency"),
            "probable_loss_stage": "retrieval_initial" if not row.get("recall@10") else "context_assembly",
            "final_context": [{"chunk_uid": uid, "text": by_uid.get(str(uid), {}).get("text", "[chunk unavailable]")} for uid in uids],
        })
    args.output.write_text(json.dumps({"source_report": str(args.report), "failures": failures}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
