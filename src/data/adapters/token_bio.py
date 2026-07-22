from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from src.data.dataset_schema import validate_record

TARGET_TYPES={"TRIỆU_CHỨNG","TÊN_XÉT_NGHIỆM","KẾT_QUẢ_XÉT_NGHIỆM","CHẨN_ĐOÁN","THUỐC"}
DEFAULT_MAPPINGS={
    "vimq":{"DRUG":"THUỐC","MEDICINE":"THUỐC","SYMPTOM_AND_DISEASE":"UNMAPPED","MEDICAL_PROCEDURE":"UNMAPPED"},
    "vietmed_ner":{"DRUG":"THUỐC","MEDICINE":"THUỐC","SYMPTOM":"TRIỆU_CHỨNG","DISEASE":"CHẨN_ĐOÁN","DIAGNOSIS":"CHẨN_ĐOÁN","LAB_TEST":"TÊN_XÉT_NGHIỆM","TEST_NAME":"TÊN_XÉT_NGHIỆM","LAB_VALUE":"KẾT_QUẢ_XÉT_NGHIỆM","TEST_RESULT":"KẾT_QUẢ_XÉT_NGHIỆM"},
    "phoner_covid19":{"SYMPTOM_AND_DISEASE":"UNMAPPED"},
}

def _split_tag(tag: str) -> tuple[str,str]:
    if tag == "O": return "O", "O"
    if "-" in tag:
        p,l=tag.split("-",1); return p,l
    return "B", tag

def load_token_bio_jsonl(path: Path, source: str, split: str, license_text: str, mapping: dict[str,str]) -> tuple[list[dict], dict]:
    rows=[]; labels=Counter(); mapped=Counter(); examples=defaultdict(list)
    for idx,line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip(): continue
        raw=json.loads(line); tokens=raw.get("tokens") or raw.get("words"); tags=raw.get("ner_tags") or raw.get("labels") or raw.get("tags")
        if tokens is None or tags is None or len(tokens) != len(tags): raise ValueError(f"token/tag schema mismatch at {path}:{idx}")
        text=" ".join(tokens); offsets=[]; pos=0
        for tok in tokens:
            offsets.append((pos,pos+len(tok))); pos += len(tok)+1
        ents=[]; cur=None; cur_label=None
        for i,tag in enumerate(tags):
            prefix,label=_split_tag(str(tag))
            if label != "O": labels[label]+=1
            target=mapping.get(label, "UNMAPPED") if label != "O" else "IGNORE"
            if label != "O": mapped[target]+=1
            if target == "IGNORE" or prefix == "O":
                if cur: ents.append(cur); cur=None; cur_label=None
                continue
            if target not in TARGET_TYPES and target != "UNMAPPED": target="UNMAPPED"
            if prefix in {"B","U"} or cur is None or cur["type"] != target or cur_label != label:
                if cur: ents.append(cur)
                s,e=offsets[i]; cur={"id":"","start":s,"end":e,"text":text[s:e],"type":target,"assertions":[],"candidates":[],"metadata":{"source_label":label}}
                cur_label=label
                if target == "UNMAPPED":
                    cur["metadata"]["needs_review"]=True
                    if len(examples[label]) < 3: examples[label].append(cur["text"])
                if prefix == "U": ents.append(cur); cur=None; cur_label=None
            else:
                cur["end"]=offsets[i][1]; cur["text"]=text[cur["start"]:cur["end"]]
        if cur: ents.append(cur)
        for j,e in enumerate(ents,1): e["id"]=f"E{j}"
        rec={"id":str(raw.get("id", f"{source}_{split}_{idx}")),"text":text,"entities":ents,"relations":[],"source":source,"source_split":split,"license":license_text,"metadata":{"upstream_id":str(raw.get("id", idx)),"synthetic":False,"schema":"token_bio"}}
        validate_record(rec); rows.append(rec)
    return rows, {"schema":"token_bio_jsonl","source_labels":dict(labels),"mapped_counts":dict(mapped),"unmapped_examples":dict(examples)}
