from __future__ import annotations
from typing import Any
VALID_TYPES={"TRIỆU_CHỨNG","TÊN_XÉT_NGHIỆM","KẾT_QUẢ_XÉT_NGHIỆM","CHẨN_ĐOÁN","THUỐC"}
VALID_ASSERTIONS={"isNegated","isFamily","isHistorical"}

def validate_record(rec: dict[str, Any]) -> None:
    text=rec["text"]; seen=set()
    for ent in rec.get("entities", []):
        s,e=ent["start"], ent["end"]
        if not isinstance(s,int) or not isinstance(e,int) or not (0 <= s < e <= len(text)): raise ValueError("invalid span")
        if text[s:e] != ent["text"]: raise ValueError(f"span invariant failed: {ent}")
        if ent["type"] not in VALID_TYPES and ent["type"] not in {"IGNORE","UNMAPPED"}: raise ValueError("invalid type")
        if any(a not in VALID_ASSERTIONS for a in ent.get("assertions", [])): raise ValueError("invalid assertion")
        for c in ent.get("candidates", []):
            if not isinstance(c.get("code", c if isinstance(c,str) else None), str): raise ValueError("candidate code must be string")
        key=(s,e,ent["text"],ent["type"])
        if key in seen: raise ValueError("duplicate entity")
        seen.add(key)
