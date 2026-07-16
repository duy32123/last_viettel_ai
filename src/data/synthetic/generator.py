from __future__ import annotations
import hashlib, json, random
from pathlib import Path
from collections import Counter, defaultdict
from src.data.dataset_schema import validate_record

DIAG=("tăng huyết áp", "I10"); DRUG=("amlodipine 10 mg po daily", "17767")

def add(parts, entities, text, typ, assertions=None, candidates=None):
    start=sum(len(p) for p in parts); parts.append(text); end=start+len(text)
    entities.append({"id":f"E{len(entities)+1}","start":start,"end":end,"text":text,"type":typ,"assertions":assertions or [],"candidates":candidates or []})

def sample(template_id:int, newline:str="\n") -> dict:
    parts=[f"Mẫu {template_id}: "]; ents=[]
    if template_id % 4 == 0:
        parts.append("Hiện tại: bệnh nhân "); add(parts, ents, "khó thở", "TRIỆU_CHỨNG"); parts.append(", không "); add(parts, ents, "đau ngực", "TRIỆU_CHỨNG", ["isNegated"]); parts.append(" nhưng còn mệt.")
    elif template_id % 4 == 1:
        parts.append("Tiền sử: "); add(parts, ents, DIAG[0], "CHẨN_ĐOÁN", ["isHistorical"], [{"code":DIAG[1],"terminology":"ICD-10","verified":False}]); parts.append(newline+"Thuốc trước nhập viện: "); add(parts, ents, DRUG[0], "THUỐC", ["isHistorical"], [{"code":DRUG[1],"terminology":"RxNorm","verified":False}])
    elif template_id % 4 == 2:
        parts.append("Mẹ bệnh nhân có "); add(parts, ents, "đái tháo đường", "CHẨN_ĐOÁN", ["isFamily"], [{"code":"E11.9","terminology":"ICD-10-CM","verified":False}]); parts.append("; BN sốt 38,5C. "); add(parts, ents, "WBC", "TÊN_XÉT_NGHIỆM"); parts.append(": "); add(parts, ents, "12,5 /mm3", "KẾT_QUẢ_XÉT_NGHIỆM")
    else:
        parts.append("- ho khan,   "); add(parts, ents, "sốt", "TRIỆU_CHỨNG"); parts.append(newline+"Xét nghiệm glucose là "); add(parts, ents, "7,2 mmol/l", "KẾT_QUẢ_XÉT_NGHIỆM")
    rec={"id":f"syn_{template_id}","text":"".join(parts),"entities":ents,"relations":[],"source":"synthetic_v1","source_split":"","license":"project-generated","metadata":{"template_family":f"family_{template_id%4}"}}
    validate_record(rec); return rec

def generate(seed:int, counts:dict, out_dir:Path):
    random.seed(seed); out_dir.mkdir(parents=True, exist_ok=True)
    splits=[]
    for split,n in counts.items():
        rows=[]
        for i in range(n):
            rec=sample(i + {"train":0,"dev":100,"test":200}[split], "\r\n" if i%2 else "\n"); rec["source_split"]=split; rows.append(rec)
        with (out_dir/f"{split}.jsonl").open("w",encoding="utf-8") as fh:
            for r in rows: fh.write(json.dumps(r,ensure_ascii=False)+"\n")
        splits += rows
    return stats(splits)

def ann_hash(r):
    norm=" ".join(r["text"].casefold().split()); anns=[(e["start"],e["end"],e["type"],tuple(e.get("assertions",[]))) for e in r["entities"]]
    return hashlib.sha256(json.dumps([norm,anns],ensure_ascii=False).encode()).hexdigest()

def assert_no_leakage(paths):
    seen={}
    for split,path in paths.items():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            r=json.loads(line); h=ann_hash(r)
            if h in seen and seen[h] != split: raise ValueError("dataset leakage detected")
            seen[h]=split

def stats(rows):
    c=Counter(); a=Counter(); s=Counter(r["source_split"] for r in rows)
    for r in rows:
        for e in r["entities"]:
            c[e["type"]]+=1
            for x in e.get("assertions",[]): a[x]+=1
    return {"records_by_split":dict(s),"entities_by_type":dict(c),"assertions":dict(a)}
