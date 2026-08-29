# Claim-level faithfulness v1

This post-generation layer is non-destructive. It receives only the completed
answer and the exact final context blocks used for generation. It does not run
retrieval, rewrite the answer, change abstention, or delay token streaming.

## Pipeline

1. Strip transport-only citation markup and split factual sentences,
   semicolon clauses, and explicit causal clauses.
2. Ignore greetings, stylistic transitions, explicit recommendations, and
   trivially attributed citation sentences.
3. Rank sentence spans from the final evidence blocks using significant-token
   coverage, with penalties for missing product references and numbers.
4. Send all claims and their top evidence spans through one batched invocation
   of the existing multilingual NLI model.
5. Combine lexical evidence, polarity, product identity, and NLI into exactly
   one of `SUPPORTED`, `PARTIALLY_SUPPORTED`, `INFERRED`, `UNSUPPORTED`, or
   `CONTRADICTED`.
6. Convert the highest-priority problem into the existing
   `PostGenerationReview` caveat channel. A fully supported answer adds no UI
   message.

Evidence contracts retain exact passage offsets plus `document_id`, primary
and fused `chunk_uid` values, path, page, section, heading path, block IDs, and
email thread ID. Trailing answer citations currently apply to every extracted
claim; sentence-local citation syntax can be added later without changing the
contracts.

## Provisional thresholds

- direct lexical support: `0.72`;
- partial support: `0.45`;
- related/inferred evidence: `0.24`;
- NLI entailment: `0.65`;
- NLI contradiction: `0.60`, accepted only with lexical overlap of at least
  `0.24` to avoid treating unrelated claims as contradictions.

These are explicit pilot defaults, not production calibration. The versioned
benchmark includes eight response scenarios and eleven claims. Both the
deterministic lower bound and runtime batched-NLI configurations are retained.
