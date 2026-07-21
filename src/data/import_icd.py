from __future__ import annotations
import csv, json
from pathlib import Path
from .kb_schema import KBRecord

def _load_rows(path: Path):
    if path.suffix.lower()==".json": return json.loads(path.read_text(encoding="utf-8"))
    delim="\t" if path.suffix.lower() in {".tsv",".tab"} else ","
    with path.open(encoding="utf-8", newline="") as fh: return list(csv.DictReader(fh, delimiter=delim))

def import_icd_official(path: Path, terminology: str, version: str, source: str, verified: bool=True, field_map: dict|None=None) -> list[KBRecord]:
    fm={"code":"code","title":"title","aliases":"aliases","parent":"parent","version":"version","language":"language", **(field_map or {})}
    out=[]
    for row in _load_rows(path):
        file_term=(row.get("terminology") or row.get("TERMINOLOGY") or terminology).strip()
        if file_term and file_term != terminology: raise ValueError(f"configured {terminology} but file contains {file_term}")
        aliases=row.get(fm["aliases"], row.get("synonyms", "")) or ""
        aliases=[a.strip() for a in aliases.split("|") if a.strip()] if isinstance(aliases,str) else list(aliases)
        ver=(row.get(fm["version"]) or version).strip()
        meta={"input_file":str(path),"parent":row.get(fm["parent"]),"official_kb":bool(verified),"source_kind":"official_local"}
        title=row.get(fm["title"], row.get("preferred_name", row.get("name", "")))
        out.append(KBRecord(row[fm["code"]].strip(), title.strip(), aliases, terminology, ver, source, verified, meta, "diagnosis", row.get(fm["language"], "vi")))
    return out

def import_icd_csv(path: Path, terminology: str, version: str, source: str, verified: bool=True) -> list[KBRecord]:
    return import_icd_official(path, terminology, version, source, verified)
