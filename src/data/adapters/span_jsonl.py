from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from src.data.dataset_schema import validate_record

TARGET_TYPES={"TRIỆU_CHỨNG","TÊN_XÉT_NGHIỆM","KẾT_QUẢ_XÉT_NGHIỆM","CHẨN_ĐOÁN","THUỐC"}

DEFAULT_MAPPINGS={
    "vimq": {
        "DRUG": "THUỐC", "drug": "THUỐC", "MEDICINE": "THUỐC", "medicine": "THUỐC",
        "SYMPTOM_AND_DISEASE": "UNMAPPED", "SYMPTOM&DISEASE": "UNMAPPED",
        "MEDICAL_PROCEDURE": "UNMAPPED", "medical_procedure": "UNMAPPED",
    },
    "vietmed_ner": {},
}


def _entity_fields(ent: Any) -> tuple[int,int,str,str|None]:
    if isinstance(ent, dict):
        s=int(ent.get("start", ent.get("s")))
        e=int(ent.get("end", ent.get("e")))
        label=str(ent.get("label", ent.get("type", ent.get("category", ent.get("c")))))
        text=ent.get("text")
        return s,e,label,text
    if isinstance(ent, (list, tuple)) and len(ent) >= 3:
        return int(ent[0]), int(ent[1]), str(ent[2]), None
    raise ValueError(f"unsupported entity shape: {ent!r}")


def load_span_jsonl(path: Path, source: str, split: str, license_text: str, mapping: dict[str,str] | None=None, synthetic: bool=False) -> tuple[list[dict], dict]:
    mapping=mapping or {}
    rows=[]; labels=Counter(); mapped=Counter(); examples=defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for idx,line in enumerate(fh):
            if not line.strip(): continue
            raw=json.loads(line)
            text=raw["text"]
            ents=[]
            for ent in raw.get("entities", []):
                s,e,label,ent_text=_entity_fields(ent)
                labels[label]+=1
                target=mapping.get(label, "needs_review")
                mapped[target]+=1
                if target in {"IGNORE"}: continue
                if target not in TARGET_TYPES and target != "UNMAPPED": target="UNMAPPED"
                actual=text[s:e]
                if ent_text is not None and actual != ent_text:
                    raise ValueError(f"offset invariant failed for {source}:{idx}:{label}: {actual!r} != {ent_text!r}")
                out={"id":f"E{len(ents)+1}","start":s,"end":e,"text":actual,"type":target,"assertions":[],"candidates":[],"metadata":{"source_label":label}}
                if target == "UNMAPPED":
                    out["metadata"]["needs_review"] = True
                    if len(examples[label]) < 3: examples[label].append(actual)
                ents.append(out)
            rec={"id":str(raw.get("id", f"{source}_{split}_{idx}")),"text":text,"entities":ents,"relations":[],"source":source,"source_split":split,"license":license_text,"metadata":{"upstream_id":str(raw.get("id", idx)),"synthetic":synthetic}}
            validate_record(rec); rows.append(rec)
    report={"source_labels":dict(labels),"mapped_counts":dict(mapped),"unmapped_examples":dict(examples)}
    return rows, report
