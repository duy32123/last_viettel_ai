from __future__ import annotations
import hashlib, re, unicodedata, difflib
from collections import Counter, defaultdict
from typing import Any, Iterable
from src.data.kb_schema import KBRecord, normalize_name
from src.linking.retrieval import LexicalIndex

ICD10_RE=re.compile(r"^[A-Z][0-9]{2}(?:\.[0-9A-Z]+)?$")
DATASET_NAME="birgermoell/icd10-clinical-notes"
NAME_FIELDS=("english_diagnosis","diagnosis_en","english_name","name_en","diagnosis","title","name","label_text","vietnamese_alias","alias")
NOTE_FIELDS=("journal_note","note","text","clinical_text")

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

def _name_items(row: dict[str,Any]) -> list[tuple[str,str]]:
    out=[]
    for f in NAME_FIELDS:
        val=str(row.get(f) or "").strip()
        if val: out.append((f,val))
    return out

def _canonical(code: str, items: list[tuple[str,str,str]]) -> str:
    en=sorted({n for lang,_,n in items if lang.startswith("en")})
    if en: return en[0]
    vi=sorted({n for lang,_,n in items if lang == "vi"})
    if vi: return vi[0]
    any_name=sorted({n for _,_,n in items}, key=lambda x:(normalize_alias(x), x))
    return any_name[0] if any_name else code

def import_auxiliary_rows(rows: Iterable[dict[str,Any]], version: str="hf-pilot", source: str=DATASET_NAME) -> tuple[list[KBRecord], dict[str,Any]]:
    total=0; invalid=[]; langs=Counter(); vi_records=0; code_rows=Counter(); name_dups=Counter(); by_code=defaultdict(list); row_keys=Counter()
    for row in rows:
        total+=1; row_key=tuple(sorted((str(k),str(v)) for k,v in row.items())); row_keys[row_key]+=1
        code=str(_get(row,"code","icd10_code","label","icd_code")).strip().upper()
        lang=str(_get(row,"language","lang", default="unknown")).strip() or "unknown"; langs[lang]+=1
        if lang == "vi": vi_records += 1
        if not valid_icd10_code(code): invalid.append(code); continue
        code_rows[code]+=1
        for field,name in _name_items(row):
            by_code[code].append((lang,field,name)); name_dups[normalize_name(name)] += 1
    records=[]; duplicate_aliases=0
    for code,items in sorted(by_code.items()):
        canonical=_canonical(code, items); rec=KBRecord(code, canonical, [], "ICD-10", version, source, False, {"source_kind":"auxiliary_hf","official_kb":False,"synthetic_notes":True,"license":"CC-BY-4.0","dataset":source,"alias_provenance":[]}, "diagnosis", "multi")
        seen=set()
        for lang,field,name in sorted(items, key=lambda x:(x[0],x[1],normalize_alias(x[2]),x[2])):
            norm=normalize_alias(name); key=(code,norm)
            if key in seen: duplicate_aliases+=1; continue
            seen.add(key)
            if name != canonical: rec.aliases.append(name)
            rec.metadata["alias_provenance"].append({"raw_alias":name,"normalized_alias":norm,"language":lang,"source_field":"name","original_field":field})
        records.append(rec)
    repeated=sum(c-1 for c in code_rows.values() if c>1)
    report={"dataset":source,"source_kind":"auxiliary_hf","official_kb":False,"synthetic_notes":True,"license":"CC-BY-4.0","records":total,"unique_codes":len(by_code),"languages":dict(langs),"vietnamese_records":vi_records,"unique_code_ratio":len(by_code)/total if total else 0.0,"dataset_unique_code_coverage":1.0 if by_code else 0.0,"repeated_multilingual_code_rows":repeated,"true_duplicate_rows":sum(c-1 for c in row_keys.values() if c>1),"duplicate_aliases":duplicate_aliases,"duplicate_names":sum(c-1 for c in name_dups.values() if c>1),"invalid_code_formats":len(invalid),"invalid_codes":invalid[:20],"verified":0,"full_icd10_coverage":False}
    return records, report

def deterministic_split_id(value: str) -> str:
    h=int(hashlib.sha256(value.encode()).hexdigest(),16)%100
    return "train" if h<80 else ("dev" if h<90 else "test")

def _note(row): return str(_get(row,*NOTE_FIELDS, default="")).strip()

def _similar_hard_negatives(note: str, code: str, codes: list[str], code_names: dict[str,str], k:int=5):
    q=normalize_alias(note); scored=[]
    for c in sorted(set(codes)):
        if c==code: continue
        scored.append((difflib.SequenceMatcher(None,q,normalize_alias(code_names.get(c,c))).ratio(),c))
    return [c for _,c in sorted(scored, key=lambda x:(-x[0],x[1]))[:k]]

def make_pilot_examples(rows: Iterable[dict[str,Any]], codes: list[str], code_names: dict[str,str]|None=None) -> dict[str,list[dict[str,Any]]]:
    out={"train":[],"dev":[],"test":[]}; code_names=code_names or {c:c for c in codes}
    seen_ids={"train":set(),"dev":set(),"test":set()}; assigned={}
    for row in rows:
        code=str(_get(row,"code","icd10_code","label","icd_code")).strip().upper(); note=_note(row)
        rid=str(_get(row,"id","original_id", default=hashlib.sha1((code+note).encode()).hexdigest()))
        if not note or not valid_icd10_code(code): continue
        split=deterministic_split_id(rid)
        if rid in assigned and assigned[rid] != split: raise ValueError(f"pilot id leakage across splits: {rid}")
        assigned[rid]=split; seen_ids[split].add(rid)
        neg=_similar_hard_negatives(note, code, codes, code_names)
        out[split].append({"id":rid,"journal_note":note,"positive_code":code,"language":str(_get(row,"language","lang", default="unknown")),"hard_negative_codes":neg,"metadata":{"official_evaluation":False,"source_kind":"auxiliary_hf","synthetic_notes":True}})
    return out

def assert_no_query_kb_leakage(records: list[KBRecord], splits: dict[str,list[dict[str,Any]]]) -> dict[str,Any]:
    kb={normalize_alias(r.canonical_name) for r in records}
    for r in records: kb.update(normalize_alias(a) for a in r.aliases)
    q=[]; split_ids={}
    for split, rows in splits.items():
        for ex in rows:
            rid=ex["id"]
            if rid in split_ids and split_ids[rid] != split: raise ValueError(f"pilot id leakage across splits: {rid}")
            split_ids[rid]=split; q.append(normalize_alias(ex["journal_note"]))
    overlap=sorted(set(q)&kb)
    if overlap: raise ValueError(f"query/KB leakage detected: {overlap[:3]}")
    return {"query_kb_overlap":0,"split_ids":{"train":sum(1 for s in split_ids.values() if s=='train'),"dev":sum(1 for s in split_ids.values() if s=='dev'),"test":sum(1 for s in split_ids.values() if s=='test')}}

def evaluate_pilot_bm25(records: list[KBRecord], splits: dict[str,list[dict[str,Any]]], top_k:int=10) -> dict[str,Any]:
    idx=LexicalIndex(records, include_unverified=True); ranks=[]; per_lang=defaultdict(list)
    for rows in splits.values():
        for ex in rows:
            cands=idx.search(ex["journal_note"], "CHẨN_ĐOÁN", top_k=max(top_k,len(records)), use_fuzzy=False)
            codes=[c.code for c in cands]; rank=(codes.index(ex["positive_code"])+1) if ex["positive_code"] in codes else None
            ranks.append(rank); per_lang[ex.get("language","unknown")].append(rank)
    def metrics(rs):
        n=len(rs) or 1
        return {"recall@1":sum(1 for r in rs if r and r<=1)/n,"recall@5":sum(1 for r in rs if r and r<=5)/n,"recall@10":sum(1 for r in rs if r and r<=10)/n,"mrr":sum(1/r for r in rs if r)/n}
    return {"official_evaluation":False,"candidate_count":len(records),"overall":metrics(ranks),"per_language":{k:metrics(v) for k,v in per_lang.items()}}
