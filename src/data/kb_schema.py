from __future__ import annotations

import hashlib, json, re, unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

VALID_TERMINOLOGIES = {"ICD-10", "ICD-10-CM", "RxNorm"}
_CODE_RE={"ICD-10": re.compile(r"^[A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?$"), "ICD-10-CM": re.compile(r"^[A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?$"), "RxNorm": re.compile(r"^[0-9]+$")}

def normalize_name(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())

@dataclass(init=False)
class KBRecord:
    code: str
    canonical_name: str
    aliases: list[str]
    terminology: str
    version: str
    semantic_type: str
    language: str
    source: str
    verified: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    normalized_names: list[str] = field(default_factory=list)

    def __init__(self, code: str, canonical_name: str, aliases: list[str]|None=None, terminology: str="", version: str="", source: str="", verified: bool=True, metadata: dict[str,Any]|None=None, semantic_type: str="", language: str="vi", terminology_version: str|None=None, preferred_name: str|None=None, synonyms: list[str]|None=None):
        self.code=str(code).strip(); self.canonical_name=(preferred_name or canonical_name or "").strip()
        self.aliases=list(aliases if aliases is not None else (synonyms or [])); self.terminology=terminology; self.version=version or (terminology_version or "")
        self.semantic_type=semantic_type or ("drug" if terminology == "RxNorm" else "diagnosis")
        self.language=language or "vi"; self.source=source; self.verified=bool(verified); self.metadata=metadata or {}; self.normalized_names=[]

    @property
    def preferred_name(self): return self.canonical_name
    @property
    def synonyms(self): return self.aliases
    @property
    def terminology_version(self): return self.version

    def validate(self) -> None:
        if not self.code: raise ValueError("KB code must be a non-empty string")
        if self.terminology not in VALID_TERMINOLOGIES: raise ValueError(f"Unsupported terminology: {self.terminology}")
        if not self.version: raise ValueError("version is required")
        if not self.canonical_name: raise ValueError("canonical_name is required")
        if not _CODE_RE[self.terminology].match(self.code): raise ValueError(f"invalid {self.terminology} code: {self.code}")
        self.aliases=sorted({a.strip() for a in self.aliases if a and a.strip() and a.strip() != self.canonical_name})
        self.normalized_names=sorted({normalize_name(n) for n in [self.canonical_name,*self.aliases] if n})

    def to_dict(self) -> dict[str,Any]:
        self.validate(); return asdict(self)
    def to_json(self) -> str: return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

def dedupe_records(records: Iterable[KBRecord]) -> list[KBRecord]:
    merged: dict[tuple[str,str,str], KBRecord]={}
    for rec in records:
        rec.validate(); key=(rec.terminology, rec.version, rec.code)
        if key not in merged: merged[key]=rec; continue
        cur=merged[key]; cur.aliases=sorted(set(cur.aliases)|set(rec.aliases)|{rec.canonical_name}); cur.metadata.setdefault("merged_sources",[]).append(rec.source); cur.verified=cur.verified and rec.verified; cur.validate()
    return sorted(merged.values(), key=lambda r:(r.terminology,r.version,r.code))

def alias_collisions(records: Iterable[KBRecord]) -> dict[str,list[str]]:
    seen: dict[tuple[str,str,str], set[str]]={}
    for r in records:
        r.validate()
        for n in r.normalized_names:
            seen.setdefault((r.terminology,r.version,n), set()).add(r.code)
    return {f"{t}|{v}|{n}": sorted(codes) for (t,v,n),codes in seen.items() if len(codes)>1}

def write_jsonl(path: Path, records: Iterable[KBRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records: fh.write(rec.to_json()+"\n")

def read_jsonl(path: Path) -> list[KBRecord]:
    rows=[]
    if not path.exists(): return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        d=json.loads(line); rows.append(KBRecord(d.get("code",""), d.get("canonical_name") or d.get("preferred_name",""), d.get("aliases") or d.get("synonyms",[]), d.get("terminology",""), d.get("version") or d.get("terminology_version",""), d.get("source",""), d.get("verified",False), d.get("metadata",{}), d.get("semantic_type",""), d.get("language","vi")))
    return rows

def file_sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""): h.update(chunk)
    return h.hexdigest()
