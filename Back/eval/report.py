from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .metrics_retrieval import average_metrics
from .schemas import BenchmarkReport, QueryEvaluation


def aggregate_by_question_type(
    results: Iterable[QueryEvaluation],
) -> Dict[str, Dict[str, Optional[float]]]:
    grouped: Dict[str, List[QueryEvaluation]] = defaultdict(list)
    for result in results:
        grouped[result.question_type].append(result)
    return {
        # ``average_metrics`` ignores None values but keeps their metric names,
        # so an all-unanswerable slice is reported explicitly with null values.
        question_type: average_metrics(item.metrics for item in items)
        for question_type, items in sorted(grouped.items())
    }


def write_report(report: BenchmarkReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
