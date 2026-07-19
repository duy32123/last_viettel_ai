from __future__ import annotations
import hashlib, json, re, unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable
from src.data.kb_schema import KBRecord, normalize_name

ICD10_RE=re.compile(r"^[A-Z][0-9]{2}(?:\.[0-9A-Z]+)?$")
DATASET_NAME="birgermoell/icd10-clinical-notes"

def normalize_alias(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text)).casefold().split())

def valid_icd10_code(code: str) -> bool:
    return bool(ICD10_RE.match(str(code or "").strip().upper()))

def _get(row: dict[str,Any], *names, default=""):
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return default

def rows_from_hf(split: str|None=None, **load_kwargs):
    from datasets import load_dataset
    ds=load_dataset(DATASET_NAME, split=split, **load_kwargs) if split else load_dataset(DATASET_NAME, **load_kwargs)
    if isinstance(ds, dict):
        for sp,part in ds.items():
            for row in part: yield {**dict(row), "_hf_split": sp}
    else:
        for row in ds: yield dict(row)

def import_auxiliary_rows(rows: Iterable[dict[str,Any]], version: str="hf-pilot", source: str=DATASET_NAME) -> tuple[list[KBRecord], dict[str,Any]]:
    by_code: dict[str,KBRecord]={}; alias_seen=set(); total=0; invalid=[]; langs=Counter(); vi_records=0; dup_alias=0; dup_codes=Counter(); names=Counter()
    for row in rows:
        total += 1
        code=str(_get(row,"code","icd10_code","label","icd_code")).strip().upper()
        lang=str(_get(row,"language","lang", default="unknown")).strip() or "unknown"; langs[lang]+=1
        if not valid_icd10_code(code): invalid.append(code); continue
        dup_codes[code]+=1
        raw_alias=str(_get(row,"vietnamese_alias","alias","diagnosis","title","name","label_text", default="")).strip()
        note=str(_get(row,"journal_note","note","text", default="")).strip()
        canonical=raw_alias or code
        rec=by_code.get(code)
        if not rec:
            rec=KBRecord(code, canonical, [], "ICD-10", version, source, False, {"source_kind":"auxiliary_hf","official_kb":False,"synthetic_notes":True,"license":"CC-BY-4.0","dataset":source}, "diagnosis", lang)
            by_code[code]=rec
        if lang == "vi":
            vi_records += 1
            for alias in [raw_alias, note]:
                if not alias: continue
                key=(code, normalize_alias(alias))
                if key in alias_seen: dup_alias += 1; continue
                alias_seen.add(key); rec.aliases.append(alias); rec.metadata.setdefault("alias_provenance",[]).append({"raw_alias":alias,"normalized_alias":key[1],"language":lang,"source_field":"journal_note" if alias==note else "alias"})
        names[normalize_name(canonical)] += 1
    records=list(by_code.values())
    report={"dataset":source,"source_kind":"auxiliary_hf","official_kb":False,"synthetic_notes":True,"license":"CC-BY-4.0","records":total,"unique_codes":len(by_code),"languages":dict(langs),"vietnamese_records":vi_records,"code_coverage":len(by_code)/total if total else 0.0,"duplicate_aliases":dup_alias,"duplicate_codes":sum(c-1 for c in dup_codes.values() if c>1),"duplicate_names":sum(c-1 for c in names.values() if c>1),"invalid_code_formats":len(invalid),"invalid_codes":invalid[:20],"verified":0,"full_icd10_coverage":False}
    return records, report

def deterministic_split_id(value: str) -> str:
    h=int(hashlib.sha256(value.encode()).hexdigest(),16)%100
    return "train" if h<80 else ("dev" if h<90 else "test")

def make_pilot_examples(rows: Iterable[dict[str,Any]], codes: list[str]) -> dict[str,list[dict[str,Any]]]:
    out={"train":[],"dev":[],"test":[]}; all_codes=sorted(set(codes))
    for row in rows:
        code=str(_get(row,"code","icd10_code","label","icd_code")).strip().upper()
        note=str(_get(row,"journal_note","note","text", default="")).strip()
        rid=str(_get(row,"id","original_id", default=hashlib.sha1((code+note).encode()).hexdigest()))
        if not note or not valid_icd10_code(code): continue
        split=deterministic_split_id(rid)
        neg=[c for c in all_codes if c!=code][:5]
        out[split].append({"id":rid,"journal_note":note,"positive_code":code,"hard_negative_codes":neg,"metadata":{"official_evaluation":False,"source_kind":"auxiliary_hf","synthetic_notes":True}})
    return out
