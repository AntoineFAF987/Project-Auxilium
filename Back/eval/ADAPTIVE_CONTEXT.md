# Adaptive context (benchmark-only)

`anchor_scope_adaptive_k` runs the unchanged retrieval/MMR/reranker and
`anchor_scope_adaptive`, then progressively selects at most `FINAL_K` evidence
chunks. It does not call an LLM and is not used by the production answer path.

## Decision policy

- A single-evidence query may stop at one chunk only when the top reranker
  probability is at least `0.5` (or has the existing exact lexical bonus) and
  is coherent through BM25+dense rank agreement or significant-query-term
  coverage.
- Comparisons require at least two chunks from two documents. Exhaustive
  requests require at least three chunks. Multi-clause questions require at
  least two chunks.
- Anchor + Scope anchors and selected scope chunks are protected until added.
- A candidate whose token Jaccard similarity with selected evidence is at
  least `0.82` is skipped, unless protected.
- Only single-evidence intents may stop after three very weak chunks, and only
  when the best available score is at most `0.15`. This limits irrelevant
  context for likely-unanswerable queries.
- Otherwise selection continues until intent and term coverage are sufficient,
  candidates are exhausted, or `FINAL_K` is reached.

The configurable defaults live in `SufficiencyPolicy`. The `0.5` reranker
boundary follows the current probability semantics. Query-term coverage
`0.55`, redundancy `0.82`, weak score `0.15`, and patience `3` are conservative
pilot defaults and must be recalibrated on a larger representative benchmark.

## Measurement

The trace records intent, policy, candidate pool, protected evidence,
redundancy skips, every decision, final decision, and sufficiency latency.
The benchmark reports final context blocks, source chunks, characters,
estimated tokens (`characters / 4`), final-k distribution, evidence coverage,
and premature stops. A stop is premature when a documentary v2 gold available
in the first `FINAL_K` post-Anchor+Scope candidates is absent from final context.
