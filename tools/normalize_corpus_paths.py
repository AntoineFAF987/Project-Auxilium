from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "Back" / "data" / "index_rag" / "corpus.jsonl"


def normalize() -> None:
    if not CORPUS.exists():
        raise SystemExit(f"Missing corpus: {CORPUS}")

    lines_out: list[str] = []
    for raw in CORPUS.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        path_value = row.get("path")
        if path_value:
            path_obj = Path(path_value)
            try:
                rel = path_obj.resolve().relative_to(ROOT.resolve())
                row["path"] = rel.as_posix()
            except Exception:
                pass
        lines_out.append(json.dumps(row, ensure_ascii=False))

    CORPUS.write_text("\n".join(lines_out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    normalize()
