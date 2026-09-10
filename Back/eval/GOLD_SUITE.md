# Gold Evaluation Suite v1

`data/gold_v1.jsonl` is the manually maintained source of truth: one readable
JSON object per case.  Every case must retain a verifiable `origin_type` and
`origin_reference`; invented benchmark questions are rejected by review.

The 50 cases cover: factual direct (8), exact entity (6), current-state and
decision (5), email/attachment (5), source constraints (5), follow-up and
transformation (5), multilingual (4), structured XLSX (4), answerability and
related evidence (4), and intelligent retry (4).

Run locally, without a provider:

```powershell
venv\Scripts\python.exe -m eval.run_gold --mode retrieval --engine offline_lexical
```

`--engine production` uses the unchanged retrieval stack and needs its local
embedding model cache. `offline_lexical` is the model-free fallback used for a
local corpus baseline; its scores must never be compared to production-run
scores. Only `execution_scope=indexed_local` rows query the current local index.
`controlled` rows are intentionally skipped there: their authoritative corpus
is the named unit-test fixture. `live_only` rows need a live source/provider.

`--mode end-to-end --live` additionally requires
`AUXILIUM_GOLD_ALLOW_LIVE=1`; this double opt-in prevents accidental provider
traffic. Its deterministic rule scorer reports required-fact recall, forbidden
claims, answerability, constraints, exact entity, current state, citations,
answer length, and latency. Retrieval mode reports document hit, context
recall, exact-entity signal, and latency. Retry success is populated only by a
trace-capable E2E adapter; v1 does not claim it from a plain retrieval result.

The observed offline-lexical baseline is document hit `0.8182`, context recall
`0.5152`, mean latency `0.37 ms` (11 index-executable cases; 39 controlled or
live-only cases skipped). Proposed thresholds for this *same offline engine*
are document hit `>= 0.80` and context recall `>= 0.50`: each is the observed
score rounded down, not a guessed product target. Production-retrieval
thresholds stay pending until its local embedding model is available and a
production-engine baseline is recorded. The non-negotiable E2E safety
thresholds are Exact Entity Violation = 0 and forbidden claim count = 0.
