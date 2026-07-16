from __future__ import annotations
import json
from pathlib import Path
from src.data.dataset_schema import validate_record

def load_jsonl(path: Path, source: str, split: str, license_text: str) -> list[dict]:
    rows=[]
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip(): continue
            rec=json.loads(line); rec.setdefault("source", source); rec.setdefault("source_split", split); rec.setdefault("license", license_text); rec.setdefault("relations", []); rec.setdefault("metadata", {})
            validate_record(rec); rows.append(rec)
    return rows
