"""Run the gold suite in retrieval-only mode, or explicitly opt in to live E2E.

The default never contacts an LLM provider.  Controlled test-derived cases are
reported as intentionally skipped by retrieval mode because their tiny corpus
is defined in their originating unit test, not in the user's index.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from .gold import aggregate, load_gold_cases, score_answer

BACK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = BACK_DIR / "eval" / "data" / "gold_v1.jsonl"


def _document_name(meta: dict[str, Any]) -> str:
    return str(meta.get("file") or Path(str(meta.get("path") or "")).name)


def _offline_lexical_retrieval(query: str, *, k: int) -> list[tuple[float, dict[str, Any]]]:
    """Dependency-free baseline over the already indexed corpus, never a prod proxy."""
    tokens = {token for token in re.findall(r"[\w-]+", query.casefold()) if len(token) > 2}
    corpus_path = BACK_DIR / "data" / "index_rag" / "corpus.jsonl"
    candidates = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        meta = json.loads(line)
        text = str(meta.get("text") or "").casefold()
        candidates.append((float(sum(token in text for token in tokens)), meta))
    return sorted(candidates, key=lambda item: item[0], reverse=True)[:k]


def run_retrieval(cases, *, k: int = 10, engine: str = "production") -> list[dict[str, Any]]:
    if engine == "offline_lexical":
        return _run_retrieval_rows(cases, k=k, search=lambda query: _offline_lexical_retrieval(query, k=k), engine=engine)
    from api.index_singleton import idx
    from rag_core.constants import FINAL_K, MAX_CONTEXT_CHARS
    from rag_core.retrieval import clip_context_blocks, fuse_contiguous_passages
    return _run_retrieval_rows(cases, k=k, search=lambda query: idx.search(query, retrieve_k=k, top_k_faiss=max(k, 20), hybrid_alpha=0.5, use_rerank=True), engine=engine, clip=lambda rows: clip_context_blocks(fuse_contiguous_passages(rows), max_chars=MAX_CONTEXT_CHARS, keep=FINAL_K))


def _run_retrieval_rows(cases, *, k: int, search, engine: str, clip=None) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        row: dict[str, Any] = {"id": case.id, "category": case.category, "scope": case.execution_scope, "engine": engine}
        if case.execution_scope != "indexed_local":
            row.update({"status": "skipped_controlled", "document_hit": None, "context_recall": None,
                        "exact_entity_violation": None, "retry_success": None, "latency_ms": None})
            rows.append(row)
            continue
        started = time.perf_counter()
        try:
            found = search(case.question)
            retrieved = found[0] if isinstance(found, tuple) else found
            context = clip(retrieved) if clip else retrieved[:min(5, len(retrieved))]
            documents = {_document_name(meta) for _, meta in retrieved[:k]}
            context_text = "\n".join(str(meta.get("text") or "") for _, meta in context)
            expected = set(case.expected_document_ids)
            required = [_matches_context(context_text, group) for group in case.required_facts]
            exact_violation = float(bool(case.exact_entities) and not all(entity.casefold() in context_text.casefold() for entity in case.exact_entities))
            row.update({"status": "ok", "document_hit": float(bool(documents & expected)),
                        "context_recall": sum(required) / len(required) if required else 1.0,
                        "exact_entity_violation": exact_violation,
                        "retry_success": None, "retrieved_documents": sorted(documents),
                        "context_document_ids": sorted({_document_name(meta) for _, meta in context})})
        except Exception as exc:
            row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "document_hit": 0.0,
                        "context_recall": 0.0, "exact_entity_violation": None, "retry_success": None})
        row["latency_ms"] = (time.perf_counter() - started) * 1000
        rows.append(row)
    return rows


def _matches_context(text: str, group: list[str]) -> bool:
    import re
    return all(re.search(term.casefold(), text.casefold()) for term in group)


def run_end_to_end(cases) -> list[dict[str, Any]]:
    """Live is deliberately a separate, explicit command path."""
    if os.environ.get("AUXILIUM_GOLD_ALLOW_LIVE") != "1":
        raise RuntimeError("live E2E requires AUXILIUM_GOLD_ALLOW_LIVE=1; it can contact the configured provider")
    from api.answer_pipeline import run_answer_pipeline
    from api.schemas import AskIn
    from starlette.requests import Request

    def request() -> Request:
        return Request({"type": "http", "method": "POST", "path": "/eval/gold", "headers": []})
    retrieval_rows = {row["id"]: row for row in run_retrieval(cases)}
    rows = []
    for case in cases:
        started = time.perf_counter()
        result = run_answer_pipeline(AskIn(q=case.question, history=case.conversation_context or []), request())
        scores = score_answer(case, result.answer, sources=result.sources, abstained=result.status == "abstained",
                              latency_ms=(time.perf_counter() - started) * 1000,
                              retrieval_trace=getattr(result, "validations", None))
        rows.append({**retrieval_rows[case.id], "status": result.status, **scores})
    return rows


def main() -> int:
    # Index initialization logs Unicode symbols; keep this evaluation command
    # usable from legacy Windows consoles without changing production logging.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mode", choices=("retrieval", "end-to-end"), default="retrieval")
    parser.add_argument("--engine", choices=("production", "production_retrieval", "offline_lexical"), default="production",
                        help="production_retrieval is strict offline, real Auxilium retrieval; lexical is never a production proxy")
    parser.add_argument("--live", action="store_true", help="required with --mode end-to-end")
    parser.add_argument("--output", type=Path, default=BACK_DIR / "eval" / "reports" / "gold-local-baseline.json")
    args = parser.parse_args()
    if args.mode == "end-to-end" and not args.live:
        parser.error("end-to-end requires --live; retrieval mode is provider-free")
    cases = load_gold_cases(args.dataset)
    if args.mode == "end-to-end":
        rows, summary = run_end_to_end(cases), None
    elif args.engine == "production_retrieval":
        from .production_retrieval import run as run_production_retrieval
        rows, summary = run_production_retrieval(cases)
    else:
        rows, summary = run_retrieval(cases, engine=args.engine), None
    payload = {"dataset": str(args.dataset), "mode": args.mode, "engine": args.engine, "case_count": len(cases), "results": rows,
               "summary": summary or aggregate(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
