# Pilot retrieval dataset

The pilot contains 20 manually verified questions over the 23 chunks currently
loaded by Auxilium: 17 answerable and 3 unanswerable.

Gold annotations follow a minimal-sufficient-evidence policy. When a reply
quotes an earlier e-mail, the original message is preferred for the earlier
fact and the reply for the update. A quoted duplicate is not added merely to
increase the number of relevant chunks. Multiple chunks are marked relevant
when distinct facts genuinely require them.

This corpus is suitable for validating the benchmark mechanics and detecting
large retrieval regressions. It is not statistically representative: it covers
only eight meaningful e-mail threads, has no non-email documents, contains
quoted-message duplication, and includes several noisy HTML chunks.

Before using the benchmark for product decisions, add independently authored
and manually adjudicated material containing at least:

- 30–50 documents per supported format and source family;
- long documents with facts beyond the first pages;
- tables, slides, spreadsheets, lists and attachments;
- semantically close distractors that differ in one decisive fact;
- temporal updates, contradictions and superseded versions;
- multi-document and multi-source questions with explicit evidence sets;
- a substantial unanswerable split, including near-miss questions;
- duplicate and quoted content with documented relevance policy.

Keep development and held-out test questions separate. At least two annotators
should adjudicate relevance grades and evidence spans for the held-out set.
