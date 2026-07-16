from __future__ import annotations

import csv
from pathlib import Path
from .kb_schema import KBRecord


def import_icd_csv(path: Path, terminology: str, version: str, source: str, verified: bool = True) -> list[KBRecord]:
    """Import a small official/organizer ICD CSV with code,name,synonyms columns."""
    records: list[KBRecord] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            synonyms = [s.strip() for s in row.get("synonyms", "").split("|") if s.strip()]
            records.append(KBRecord(row["code"].strip(), row["preferred_name"].strip(), synonyms, terminology, version, source, verified, {"input_file": str(path)}))
    return records
