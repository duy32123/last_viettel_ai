from __future__ import annotations

import csv
from pathlib import Path
from .kb_schema import KBRecord


def import_rxnorm_csv(path: Path, version: str, source: str, verified: bool = True) -> list[KBRecord]:
    """Import RxNorm CSV with code/RXCUI, preferred_name, synonyms columns."""
    records: list[KBRecord] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            code = (row.get("code") or row.get("rxcui") or row.get("RXCUI") or "").strip()
            synonyms = [s.strip() for s in row.get("synonyms", "").split("|") if s.strip()]
            records.append(KBRecord(code, row["preferred_name"].strip(), synonyms, "RxNorm", version, source, verified, {"input_file": str(path)}))
    return records
