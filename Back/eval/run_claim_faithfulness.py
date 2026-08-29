from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Dict, List

from eval.ablation import percentile
from rag_core.faithfulness import ClaimStatus, verify_answer_claims


BACK_DIR = Path(__file__).resolve().parent.parent


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BACK_DIR / path


def run(config_path: Path, output_override: Path | None = None) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_path = _resolve(config["dataset"]).resolve()
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    use_nli = bool(config.get("use_nli", False))

    results: List[Dict[str, Any]] = []
    supported_tp = predicted_supported = 0
    problem_tp = expected_problems = 0
    false_positives = false_negatives = 0
    mapping_correct = mapping_total = 0
    unnecessary_caveats = missing_caveats = 0
    latencies: List[float] = []

    for row in rows:
        started = perf_counter()
        review = verify_answer_claims(
            row["answer"],
            row["evidence"],
            cited_source_indices=row.get("cited_source_indices") or [],
            use_nli=use_nli,
        )
        elapsed_ms = (perf_counter() - started) * 1000.0
        latencies.append(elapsed_ms)
        expected_statuses = row["expected_statuses"]
        expected_citations = row.get("expected_citation_correct") or [None] * len(expected_statuses)
        expected_sources = row.get("expected_sources") or [None] * len(expected_statuses)
        claim_rows = []
        for index, verification in enumerate(review.claims):
            expected_status = expected_statuses[index]
            expected_citation = expected_citations[index] if index < len(expected_citations) else None
            predicted_problem = (
                verification.status != ClaimStatus.SUPPORTED
                or verification.citation_correct is False
            )
            expected_problem = expected_status != ClaimStatus.SUPPORTED.value or expected_citation is False
            if verification.status == ClaimStatus.SUPPORTED:
                predicted_supported += 1
                supported_tp += int(expected_status == ClaimStatus.SUPPORTED.value)
            if expected_problem:
                expected_problems += 1
                problem_tp += int(predicted_problem)
                false_negatives += int(not predicted_problem)
            else:
                false_positives += int(predicted_problem)
            expected_source = expected_sources[index] if index < len(expected_sources) else None
            predicted_source = verification.evidence[0].source_index if verification.evidence else None
            if expected_source is not None:
                mapping_total += 1
                mapping_correct += int(predicted_source == expected_source)
            claim_rows.append({
                "text": verification.claim.text,
                "expected_status": expected_status,
                "predicted_status": verification.status.value,
                "expected_citation_correct": expected_citation,
                "predicted_citation_correct": verification.citation_correct,
                "expected_source": expected_source,
                "predicted_source": predicted_source,
                "reason": verification.reason,
            })
        expected_caveat = bool(row["expected_caveat"])
        unnecessary_caveats += int(review.caveat_required and not expected_caveat)
        missing_caveats += int(expected_caveat and not review.caveat_required)
        results.append({
            "case_id": row["case_id"],
            "expected_caveat": expected_caveat,
            "predicted_caveat": review.caveat_required,
            "latency_ms": elapsed_ms,
            "review": review.to_dict(),
            "claims": claim_rows,
        })

    claim_count = sum(len(item["claims"]) for item in results)
    report = {
        "schema_version": 1,
        "benchmark_name": config.get("benchmark_name"),
        "configuration": {**config, "dataset": str(dataset_path)},
        "case_count": len(results),
        "claim_count": claim_count,
        "metrics": {
            "supported_precision": supported_tp / predicted_supported if predicted_supported else 0.0,
            "problematic_claim_recall": problem_tp / expected_problems if expected_problems else 0.0,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "status_accuracy": (
                sum(
                    claim["expected_status"] == claim["predicted_status"]
                    for result in results for claim in result["claims"]
                ) / claim_count if claim_count else 0.0
            ),
            "evidence_mapping_accuracy": mapping_correct / mapping_total if mapping_total else 0.0,
            "unnecessary_caveat_rate": unnecessary_caveats / len(results) if results else 0.0,
            "missing_caveat_rate": missing_caveats / len(results) if results else 0.0,
            "claims_per_response_mean": claim_count / len(results) if results else 0.0,
            "evidence_chunks_inspected_mean": mean(
                item["review"]["evidence_chunks_inspected"] for item in results
            ) if results else 0.0,
            "model_calls_total": sum(item["review"]["model_calls"] for item in results),
            "latency_mean_ms": mean(latencies) if latencies else 0.0,
            "latency_p95_ms": percentile(latencies, 95) or 0.0,
            "extraction_mean_ms": mean(
                item["review"]["extraction_ms"] for item in results
            ) if results else 0.0,
            "verification_mean_ms": mean(
                item["review"]["verification_ms"] for item in results
            ) if results else 0.0,
        },
        "results": results,
    }
    if output_override:
        output_path = output_override
    else:
        output_dir = _resolve(config.get("output_dir", "eval/reports"))
        output_path = output_dir / "claim-faithfulness-v1.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the versioned claim-level faithfulness benchmark.")
    parser.add_argument("--config", default="eval/configs/claim_faithfulness.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = run(_resolve(args.config).resolve(), Path(args.output).resolve() if args.output else None)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
