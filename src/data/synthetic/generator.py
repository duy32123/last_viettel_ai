from __future__ import annotations
import hashlib, json, random, re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any
from src.data.dataset_schema import validate_record

TEMPLATE_FAMILIES = [
    "current_symptoms", "history_drug", "family_labs", "resp_labs", "mixed_followup", "rx_change",
    "abbrev_vitals", "no_diacritics", "typo_controlled", "case_punct", "dose_route", "neg_scope",
    "multi_problem", "family_history", "lab_panel",
]


def _load_kb(paths: dict | None) -> dict[str, dict[str, dict[str, Any]]]:
    kb={"diagnosis":{}, "drug":{}}
    if not paths: return kb
    for kind, key in [("diagnosis", "diagnosis"), ("drug", "drug")]:
        for p in paths.get(key, []):
            path=Path(p)
            if not path.exists(): continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                r=json.loads(line)
                cand={"code":r["code"], "terminology":r["terminology"], "verified":bool(r.get("verified", False))}
                if cand["code"] == "UNVERIFIED":
                    raise ValueError(f"KB contains forbidden fake candidate code for {r.get('preferred_name')}")
                keys={r["preferred_name"].casefold(), *[n.casefold() for n in r.get("normalized_names", [])], *[s.casefold() for s in r.get("synonyms", [])]}
                for k in keys: kb[kind][k]=cand
    return kb


def _cand(kb, kind, name, strict: bool):
    cand = kb.get(kind, {}).get(name.casefold())
    if cand: return [dict(cand)], {}
    if strict: raise ValueError(f"missing KB candidate for mention: {name}")
    return [], {"candidate_missing": True}


def add(parts, entities, text, typ, assertions=None, candidates=None, metadata=None):
    start=sum(len(p) for p in parts); parts.append(text); end=start+len(text)
    ent={"id":f"E{len(entities)+1}","start":start,"end":end,"text":text,"type":typ,"assertions":assertions or [],"candidates":candidates or []}
    if metadata: ent["metadata"]=metadata
    entities.append(ent)


def _add_diag(parts, ents, text, assertions, kb, strict):
    c,m=_cand(kb,"diagnosis",text,strict); add(parts, ents, text, "CHẨN_ĐOÁN", assertions, c, m)


def _add_drug(parts, ents, text, assertions, kb, strict):
    lookup=text.split()[0] if " " in text else text
    c,m=_cand(kb,"drug",lookup,strict); add(parts, ents, text, "THUỐC", assertions, c, m)


def sample(template_id:int, family:str, kb:dict | None=None, newline:str="\n", strict_candidates: bool=False) -> dict:
    kb=kb or {"diagnosis":{},"drug":{}}
    parts=[f"Mẫu {template_id}: "]; ents=[]
    if family == "current_symptoms":
        parts.append("Hiện tại: bệnh nhân "); add(parts, ents, "khó thở", "TRIỆU_CHỨNG"); parts.append(", không "); add(parts, ents, "đau ngực", "TRIỆU_CHỨNG", ["isNegated"]); parts.append(" nhưng còn "); add(parts, ents, "mệt", "TRIỆU_CHỨNG"); parts.append(".")
    elif family == "history_drug":
        parts.append("Tiền sử: "); _add_diag(parts, ents, "tăng huyết áp", ["isHistorical"], kb, strict_candidates); parts.append(newline+"Thuốc trước nhập viện: "); _add_drug(parts, ents, "amlodipine", ["isHistorical"], kb, strict_candidates); parts.append(" 10 mg po daily")
    elif family == "family_labs":
        parts.append("Mẹ bệnh nhân có "); _add_diag(parts, ents, "đái tháo đường", ["isFamily"], kb, strict_candidates); parts.append("; BN "); add(parts, ents, "sốt", "TRIỆU_CHỨNG"); parts.append(" 38,5C. "); add(parts, ents, "WBC", "TÊN_XÉT_NGHIỆM"); parts.append(": "); add(parts, ents, "12,5 /mm3", "KẾT_QUẢ_XÉT_NGHIỆM")
    elif family == "resp_labs":
        parts.append("- "); add(parts, ents, "ho khan", "TRIỆU_CHỨNG"); parts.append(",   "); add(parts, ents, "sốt", "TRIỆU_CHỨNG"); parts.append(newline+"Xét nghiệm "); add(parts, ents, "glucose", "TÊN_XÉT_NGHIỆM"); parts.append(" là "); add(parts, ents, "7,2 mmol/l", "KẾT_QUẢ_XÉT_NGHIỆM")
    elif family == "mixed_followup":
        parts.append("Tái khám vì "); add(parts, ents, "mệt", "TRIỆU_CHỨNG"); parts.append(", không "); add(parts, ents, "sốt", "TRIỆU_CHỨNG", ["isNegated"]); parts.append(" sau điều trị "); _add_diag(parts, ents, "đái tháo đường", [], kb, strict_candidates)
    elif family == "rx_change":
        parts.append("Đổi thuốc từ "); _add_drug(parts, ents, "amlodipine", [], kb, strict_candidates); parts.append(" do còn "); add(parts, ents, "ho khan", "TRIỆU_CHỨNG")
    elif family == "abbrev_vitals":
        parts.append("BN "); add(parts, ents, "khó thở", "TRIỆU_CHỨNG"); parts.append(", XN "); add(parts, ents, "glucose", "TÊN_XÉT_NGHIỆM"); parts.append("="); add(parts, ents, "8.1 mmol/L", "KẾT_QUẢ_XÉT_NGHIỆM")
    elif family == "no_diacritics":
        parts.append("Benh nhan "); add(parts, ents, "ho khan", "TRIỆU_CHỨNG"); parts.append(" va "); add(parts, ents, "met", "TRIỆU_CHỨNG")
    elif family == "typo_controlled":
        parts.append("Ghi nhận "); add(parts, ents, "sôt", "TRIỆU_CHỨNG"); parts.append(", "); add(parts, ents, "ho khan", "TRIỆU_CHỨNG"); parts.append(" nhẹ")
    elif family == "case_punct":
        parts.append("CHẨN ĐOÁN: "); _add_diag(parts, ents, "tăng huyết áp", [], kb, strict_candidates); parts.append("; triệu chứng: "); add(parts, ents, "MỆT", "TRIỆU_CHỨNG")
    elif family == "dose_route":
        parts.append("Dùng "); _add_drug(parts, ents, "amlodipine 5mg uống sáng", [], kb, strict_candidates); parts.append("; còn "); add(parts, ents, "đau ngực", "TRIỆU_CHỨNG")
    elif family == "neg_scope":
        parts.append("Không ghi nhận "); add(parts, ents, "sốt", "TRIỆU_CHỨNG", ["isNegated"]); parts.append(" hay "); add(parts, ents, "ho khan", "TRIỆU_CHỨNG", ["isNegated"]); parts.append(", nhưng còn "); add(parts, ents, "mệt", "TRIỆU_CHỨNG")
    elif family == "multi_problem":
        _add_diag(parts, ents, "đái tháo đường", [], kb, strict_candidates); parts.append(" kèm "); _add_diag(parts, ents, "tăng huyết áp", [], kb, strict_candidates); parts.append(", xét nghiệm "); add(parts, ents, "WBC", "TÊN_XÉT_NGHIỆM"); parts.append(" "); add(parts, ents, "10 G/L", "KẾT_QUẢ_XÉT_NGHIỆM")
    elif family == "family_history":
        parts.append("Bố có "); _add_diag(parts, ents, "tăng huyết áp", ["isFamily"], kb, strict_candidates); parts.append(", bản thân từng "); _add_diag(parts, ents, "đái tháo đường", ["isHistorical"], kb, strict_candidates)
    else:
        parts.append("XN: "); add(parts, ents, "glucose", "TÊN_XÉT_NGHIỆM"); parts.append(" "); add(parts, ents, "6,4 mmol/l", "KẾT_QUẢ_XÉT_NGHIỆM"); parts.append("; "); add(parts, ents, "WBC", "TÊN_XÉT_NGHIỆM"); parts.append(" "); add(parts, ents, "7 G/L", "KẾT_QUẢ_XÉT_NGHIỆM")
    rec={"id":f"syn_{template_id}","text":"".join(parts),"entities":ents,"relations":[],"source":"synthetic_v1","source_split":"","license":"project-generated","metadata":{"template_family":family,"gold_evaluation":False}}
    validate_record(rec); return rec


def split_families(seed:int, splits):
    fams=TEMPLATE_FAMILIES[:]; random.Random(seed).shuffle(fams)
    return {s: fams[i::len(splits)] for i,s in enumerate(splits)}


def ann_hash(r):
    text=re.sub(r"^Mẫu\s+\d+:\s*", "", r["text"])
    norm=" ".join(text.casefold().split())
    spans=[(e["text"].casefold(),e["type"],tuple(e.get("assertions",[]))) for e in r["entities"]]
    return hashlib.sha256(json.dumps([norm,spans],ensure_ascii=False).encode()).hexdigest()


def generate(seed:int, counts:dict, out_dir:Path, kb_paths:dict | None=None, strict_candidates: bool=False):
    rng=random.Random(seed); out_dir.mkdir(parents=True, exist_ok=True); kb=_load_kb(kb_paths)
    fam_by_split=split_families(seed, list(counts.keys()))
    all_rows=[]
    for split,n in counts.items():
        rows=[]; seen=set(); fams=fam_by_split[split]; attempts=0
        while len(rows) < n:
            attempts += 1
            if attempts > max(100, n*20): raise ValueError(f"unable to generate {n} unique records for split {split}")
            fam=fams[(len(rows)+attempts) % len(fams)]; tid=rng.randrange(1_000_000)
            rec=sample(tid, fam, kb, "\r\n" if rng.randrange(2) else "\n", strict_candidates)
            suffixes=["", " Ghi chú: theo dõi.", " Khuyến cáo tái khám!", " -- ổn định.", " (đã báo BS)"]
            suffix=suffixes[rng.randrange(len(suffixes))]
            rec["text"] += suffix; rec["metadata"]["variant_suffix"] = suffix; validate_record(rec)
            rec["source_split"]=split
            h=ann_hash(rec)
            if h in seen: continue
            seen.add(h); rows.append(rec)
        with (out_dir/f"{split}.jsonl").open("w",encoding="utf-8") as fh:
            for r in rows: fh.write(json.dumps(r,ensure_ascii=False)+"\n")
        all_rows += rows
    return stats(all_rows)


def assert_no_leakage(paths):
    seen={}; families={}
    for split,path in paths.items():
        families[split]=set()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            r=json.loads(line); families[split].add(r["metadata"]["template_family"]); h=ann_hash(r)
            if h in seen and seen[h] != split: raise ValueError("dataset leakage detected")
            seen[h]=split
    vals=list(families.items())
    for i,(a,fa) in enumerate(vals):
        for b,fb in vals[i+1:]:
            if fa & fb: raise ValueError(f"template_family leakage between {a} and {b}: {fa & fb}")


def stats(rows):
    c=Counter(); a=Counter(); s=Counter(r["source_split"] for r in rows)
    for r in rows:
        for e in r["entities"]:
            c[e["type"]]+=1
            for x in e.get("assertions",[]): a[x]+=1
    return {"records_by_split":dict(s),"entities_by_type":dict(c),"assertions":dict(a)}
