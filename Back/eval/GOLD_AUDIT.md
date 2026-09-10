# Audit des matériaux réutilisés

| Domaine | Sources auditées et réemployées | Couverture v1 | Manque restant |
|---|---|---|---|
| Génération gelée | `eval/generation_frozen_context.py`, `tests/eval/test_generation_frozen_context.py` | 3730/3731, état courant, prix KG2, patterns réponse | Plus de contextes gelés issus de documents non-email |
| Exact Entity / answerability | `tests/rag_core/test_answerability.py`, `tests/api/test_variant_aware_evidence.py` | 3730/3731, HV01/HV02, KG2/KG9, DN50/DN80, 2420/2422 | Preuves réelles indexées pour chaque paire voisine |
| Current state / décision | `tests/api/test_iterative_retrieval.py`, `tests/api/test_intelligent_retry.py` | pending/approved, ordre chronologique, état final | Cas réels rejetés/supersédés annotés |
| Planner, emails, PJ | `tests/api/test_source_planner.py`, `tests/api/test_answer_pipeline.py` | header/body, parent→PJ, hard source constraints, transformations | Corpus indexé de PJ textuelles suffisamment large |
| XLSX structuré | `tests/rag_core/test_structured_xlsx.py`, `tests/api/test_source_planner.py` | achat/vente, KG2, DN50, formule | Classeur réel indexé et stable pour benchmark retrieval |
| Multilingue | `tests/api/test_variant_aware_evidence.py`, `tests/rag_core/test_answerability.py` | FR→EN, prédicat précis, décision | Documents réels bilingues avec jugements de chunks |
| Corpus réel | `eval/data/pilot.jsonl`, `data/emails_flattened/2026-06-01__*.txt` | 10 cas `indexed_local` sur e-mails API, Kubernetes, PostgreSQL, Redis, dashboard, sécurité, Microsoft | PDF, local files et XLSX réellement présents dans l’index actuel |

Les 40 cas `controlled` ne sont pas des questions inventées : ils sont
directement dérivés du test nommé dans `origin_reference`. Ils ne sont pas
exécutés sur l’index utilisateur car celui-ci ne contient pas leurs micro-
corpus de test; ils restent exécutables via leurs tests unitaires et prêts pour
un adaptateur de corpus contrôlé ultérieur.
