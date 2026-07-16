from __future__ import annotations
import hashlib, json, random, re
from pathlib import Path
from collections import Counter
from src.data.dataset_schema import validate_record

TEMPLATE_FAMILIES = ["current_symptoms", "history_drug", "family_labs", "resp_labs", "mixed_followup", "rx_change"]


def _load_kb(paths: dict | None) -> dict[str, dict[str, tuple[str, str]]]:
    kb={"diagnosis":{}, "drug":{}}
    if not paths: return kb
    for kind, key in [("diagnosis", "diagnosis"), ("drug", "drug")]:
        for p in paths.get(key, []):
            path=Path(p)
            if not path.exists(): continue
            for line in path.read_text(encoding="utf-8").splitlines():
                r=json.loads(line); kb[kind][r["preferred_name"].casefold()]=(r["code"], r["terminology"])
                for n in r.get("normalized_names", []): kb[kind][n]=(r["code"], r["terminology"])
    return kb


def _cand(kb, kind, name, fallback_term):
    code, term = kb.get(kind, {}).get(name.casefold(), ("UNVERIFIED", fallback_term))
    return [{"code":code,"terminology":term,"verified":False}]


def add(parts, entities, text, typ, assertions=None, candidates=None):
    start=sum(len(p) for p in parts); parts.append(text); end=start+len(text)
    entities.append({"id":f"E{len(entities)+1}","start":start,"end":end,"text":text,"type":typ,"assertions":assertions or [],"candidates":candidates or []})


def sample(template_id:int, family:str, kb:dict | None=None, newline:str="\n") -> dict:
    kb=kb or {"diagnosis":{},"drug":{}}
    parts=[f"Mẫu {template_id}: "]; ents=[]
    if family == "current_symptoms":
        parts.append("Hiện tại: bệnh nhân "); add(parts, ents, "khó thở", "TRIỆU_CHỨNG"); parts.append(", không "); add(parts, ents, "đau ngực", "TRIỆU_CHỨNG", ["isNegated"]); parts.append(" nhưng còn "); add(parts, ents, "mệt", "TRIỆU_CHỨNG"); parts.append(".")
    elif family == "history_drug":
        parts.append("Tiền sử: "); add(parts, ents, "tăng huyết áp", "CHẨN_ĐOÁN", ["isHistorical"], _cand(kb,"diagnosis","tăng huyết áp","ICD-10")); parts.append(newline+"Thuốc trước nhập viện: "); add(parts, ents, "amlodipine", "THUỐC", ["isHistorical"], _cand(kb,"drug","amlodipine","RxNorm")); parts.append(" 10 mg po daily")
    elif family == "family_labs":
        parts.append("Mẹ bệnh nhân có "); add(parts, ents, "đái tháo đường", "CHẨN_ĐOÁN", ["isFamily"], _cand(kb,"diagnosis","đái tháo đường","ICD-10")); parts.append("; BN "); add(parts, ents, "sốt", "TRIỆU_CHỨNG"); parts.append(" 38,5C. "); add(parts, ents, "WBC", "TÊN_XÉT_NGHIỆM"); parts.append(": "); add(parts, ents, "12,5 /mm3", "KẾT_QUẢ_XÉT_NGHIỆM")
    elif family == "resp_labs":
        parts.append("- "); add(parts, ents, "ho khan", "TRIỆU_CHỨNG"); parts.append(",   "); add(parts, ents, "sốt", "TRIỆU_CHỨNG"); parts.append(newline+"Xét nghiệm "); add(parts, ents, "glucose", "TÊN_XÉT_NGHIỆM"); parts.append(" là "); add(parts, ents, "7,2 mmol/l", "KẾT_QUẢ_XÉT_NGHIỆM")
    elif family == "mixed_followup":
        parts.append("Tái khám vì "); add(parts, ents, "mệt", "TRIỆU_CHỨNG"); parts.append(", không "); add(parts, ents, "sốt", "TRIỆU_CHỨNG", ["isNegated"]); parts.append(" sau điều trị "); add(parts, ents, "đái tháo đường", "CHẨN_ĐOÁN", [], _cand(kb,"diagnosis","đái tháo đường","ICD-10"))
    else:
        parts.append("Đổi thuốc từ "); add(parts, ents, "amlodipine", "THUỐC", [], _cand(kb,"drug","amlodipine","RxNorm")); parts.append(" do còn "); add(parts, ents, "ho khan", "TRIỆU_CHỨNG")
    rec={"id":f"syn_{template_id}","text":"".join(parts),"entities":ents,"relations":[],"source":"synthetic_v1","source_split":"","license":"project-generated","metadata":{"template_family":family}}
    validate_record(rec); return rec


def split_families(seed:int, splits):
    fams=TEMPLATE_FAMILIES[:]; random.Random(seed).shuffle(fams)
    return {s: fams[i::len(splits)] for i,s in enumerate(splits)}


def generate(seed:int, counts:dict, out_dir:Path, kb_paths:dict | None=None):
    rng=random.Random(seed); out_dir.mkdir(parents=True, exist_ok=True); kb=_load_kb(kb_paths)
    fam_by_split=split_families(seed, list(counts.keys()))
    all_rows=[]
    for split,n in counts.items():
        rows=[]; fams=fam_by_split[split]
        for i in range(n):
            fam=fams[i % len(fams)]; tid=rng.randrange(1_000_000); rec=sample(tid, fam, kb, "\r\n" if rng.randrange(2) else "\n"); rec["source_split"]=split; rows.append(rec)
        with (out_dir/f"{split}.jsonl").open("w",encoding="utf-8") as fh:
            for r in rows: fh.write(json.dumps(r,ensure_ascii=False)+"\n")
        all_rows += rows
    return stats(all_rows)


def ann_hash(r):
    text=re.sub(r"^Mẫu\s+\d+:\s*", "", r["text"])
    norm=" ".join(text.casefold().split()); spans=[]
    for e in r["entities"]:
        spans.append((e["text"].casefold(),e["type"],tuple(e.get("assertions",[]))))
    return hashlib.sha256(json.dumps([norm,spans],ensure_ascii=False).encode()).hexdigest()


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
