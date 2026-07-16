from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

VALID_TERMINOLOGIES = {"ICD-10", "ICD-10-CM", "RxNorm"}


def normalize_name(text: str) -> str:
    """Normalize names for retrieval/dedup support, not for offsets."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


@dataclass
class KBRecord:
    code: str
    preferred_name: str
    synonyms: list[str]
    terminology: str
    terminology_version: str
    source: str
    verified: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    normalized_names: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.code or not isinstance(self.code, str):
            raise ValueError("KB code must be a non-empty string")
        if self.terminology not in VALID_TERMINOLOGIES:
            raise ValueError(f"Unsupported terminology: {self.terminology}")
        if not self.terminology_version:
            raise ValueError("terminology_version is required")
        if not self.preferred_name:
            raise ValueError("preferred_name is required")
        self.synonyms = sorted({s for s in self.synonyms if s and s != self.preferred_name})
        names = [self.preferred_name, *self.synonyms]
        self.normalized_names = sorted({normalize_name(n) for n in names if n})

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def dedupe_records(records: Iterable[KBRecord]) -> list[KBRecord]:
    merged: dict[tuple[str, str, str], KBRecord] = {}
    for rec in records:
        rec.validate()
        key = (rec.terminology, rec.terminology_version, rec.code)
        if key not in merged:
            merged[key] = rec
            continue
        cur = merged[key]
        cur.synonyms = sorted(set(cur.synonyms) | set(rec.synonyms) | {rec.preferred_name})
        cur.metadata.setdefault("merged_sources", []).append(rec.source)
        cur.verified = cur.verified and rec.verified
        cur.validate()
    return sorted(merged.values(), key=lambda r: (r.terminology, r.code))


def write_jsonl(path: Path, records: Iterable[KBRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
