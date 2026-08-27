from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Set

from .schemas import BenchmarkQuery


class DatasetError(ValueError):
    pass


def load_jsonl(path: str | Path) -> List[BenchmarkQuery]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise DatasetError(f"Benchmark dataset not found: {dataset_path}")

    rows: List[BenchmarkQuery] = []
    seen_ids = set()
    for line_number, raw_line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            row = BenchmarkQuery.model_validate(payload)
        except Exception as exc:
            raise DatasetError(
                f"Invalid benchmark row at {dataset_path}:{line_number}: {exc}"
            ) from exc
        if row.query_id in seen_ids:
            raise DatasetError(f"Duplicate query_id at line {line_number}: {row.query_id}")
        seen_ids.add(row.query_id)
        rows.append(row)

    if not rows:
        raise DatasetError(f"Benchmark dataset is empty: {dataset_path}")
    return rows


def write_jsonl(path: str | Path, rows: Iterable[BenchmarkQuery]) -> None:
    """Utility for dataset curation; the benchmark runner only reads datasets."""

    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(row.model_dump_json() for row in rows)
    dataset_path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def validate_relevant_chunk_ids(
    rows: Iterable[BenchmarkQuery], available_chunk_ids: Iterable[str]
) -> None:
    """Fail fast when a gold annotation does not exist in the loaded index."""

    available: Set[str] = set(available_chunk_ids)
    missing = {
        chunk_id
        for row in rows
        for chunk_id in row.relevant_chunk_ids
        if chunk_id not in available
    }
    if missing:
        raise DatasetError(
            "Gold chunk IDs absent from the current index: "
            + ", ".join(sorted(missing))
        )
