# -*- coding: utf-8 -*-
"""
Orchestration "pré-LLM" & helpers — inchangé.
"""
from typing import List, Tuple, Dict
from pathlib import Path
import numpy as np
from .constants import MAX_CONTEXT_CHARS, FINAL_K, FUSE_ADJACENT_GAP, ANSWER_MIN_CE, HISTORY_MAX_TURNS

def fuse_contiguous_passages(items: List[Tuple[float, Dict]], gap: int = FUSE_ADJACENT_GAP) -> List[Tuple[float, Dict]]:
    if not items: return []
    grouped: Dict[str, List[Tuple[float, Dict]]] = {}
    for s, m in items:
        grouped.setdefault(m.get("document_id") or m["path"], []).append((s, m))
    fused = []
    for path, arr in grouped.items():
        arr.sort(key=lambda x: x[1].get("order", x[1]["chunk_id"]))
        buf_text, buf_meta = "", None
        buf_scores = []
        prev_chunk = None
        prev_uid = None
        buf_uids = []
        for s, m in arr:
            explicit_structure = bool(m.get("chunk_uid"))
            same_zone = (
                m.get("section_id") == (buf_meta or {}).get("section_id")
                and m.get("page") == (buf_meta or {}).get("page")
            )
            linked_neighbor = bool(
                prev_uid and m.get("previous_chunk_uid") == prev_uid
            )
            legacy_neighbor = (
                not explicit_structure and prev_chunk is not None
                and m["chunk_id"] - prev_chunk <= gap
            )
            if prev_chunk is not None and ((linked_neighbor and same_zone) or legacy_neighbor):
                buf_text += ("\n\n" + m["text"]); buf_scores.append(s); prev_chunk = m["chunk_id"]; prev_uid = m.get("chunk_uid"); buf_uids.append(m.get("chunk_uid"))
            else:
                if buf_meta is not None:
                    avg_s = float(np.mean(buf_scores)) if buf_scores else 0.0
                    fused.append((avg_s, {**buf_meta, "text": buf_text, "fused_chunk_uids": [uid for uid in buf_uids if uid]}))
                buf_meta = {k: v for k, v in m.items() if k != "text"}
                buf_text = m["text"]; buf_scores = [s]; prev_chunk = m["chunk_id"]; prev_uid = m.get("chunk_uid"); buf_uids = [m.get("chunk_uid")]
        if buf_meta is not None:
            avg_s = float(np.mean(buf_scores)) if buf_scores else 0.0
            fused.append((avg_s, {**buf_meta, "text": buf_text, "fused_chunk_uids": [uid for uid in buf_uids if uid]}))
    fused.sort(key=lambda x: x[0], reverse=True)
    return fused

def clip_context_blocks(blocks: List[Tuple[float, Dict]], max_chars: int = MAX_CONTEXT_CHARS, keep: int = FINAL_K) -> List[Tuple[float, Dict]]:
    if not blocks: return []
    blocks = blocks[:keep]
    total = sum(len(b[1]["text"]) for b in blocks)
    if total <= max_chars: return blocks
    ratio = max_chars / (total + 1e-9)
    clipped = []
    for s, m in blocks:
        txt = m["text"]; new_len = max(500, int(len(txt) * ratio))
        clipped.append((s, {**m, "text": txt[:new_len]}))
    return clipped

def format_context_for_llm(blocks: List[Tuple[float, Dict]]) -> str:
    lines = []
    for i, (s, m) in enumerate(blocks, 1):
        location = []
        if m.get("page") is not None:
            location.append(f"page {m['page']}")
        if m.get("section"):
            location.append(f"section {m['section']}")
        suffix = " | " + " | ".join(location) if location else ""
        src = f"{m['file']} | chunk {m['chunk_id']} | {Path(m['path']).name}{suffix}"
        lines.append(f"[{i}] Source: {src}")
        lines.append(m["text"].strip()); lines.append("")
    return "\n".join(lines).strip()

def answerability_guard(ce_scores: List[float], threshold: float = ANSWER_MIN_CE) -> bool:
    if not ce_scores: return True
    top = max(ce_scores); return top >= threshold

def trim_history(messages, max_turns: int = HISTORY_MAX_TURNS):
    return messages[-(2*max_turns):]
