from __future__ import annotations
from typing import Any
VALID_TYPES={"TRIỆU_CHỨNG","TÊN_XÉT_NGHIỆM","KẾT_QUẢ_XÉT_NGHIỆM","CHẨN_ĐOÁN","THUỐC"}
VALID_ASSERTIONS={"isNegated","isFamily","isHistorical"}
VALID_TERMINOLOGIES={"ICD-10","ICD-10-CM","RxNorm"}


def validate_candidate(c: Any) -> None:
    if isinstance(c, str):
        if not c: raise ValueError("candidate code must be non-empty string")
        return
    if not isinstance(c, dict): raise ValueError("candidate must be string or object")
    if not isinstance(c.get("code"), str) or not c["code"]: raise ValueError("candidate code must be non-empty string")
    if c.get("terminology") not in VALID_TERMINOLOGIES: raise ValueError("candidate terminology is invalid")
    if not isinstance(c.get("verified"), bool): raise ValueError("candidate verified must be boolean")


def serialize_competition_candidates(ent: dict[str, Any]) -> list[str]:
    out=[]
    for c in ent.get("candidates", []):
        out.append(c if isinstance(c, str) else c["code"])
    return out


def validate_record(rec: dict[str, Any]) -> None:
    text=rec["text"]; seen=set()
    for ent in rec.get("entities", []):
        s,e=ent["start"], ent["end"]
        if not isinstance(s,int) or not isinstance(e,int) or not (0 <= s < e <= len(text)): raise ValueError("invalid span")
        if text[s:e] != ent["text"]: raise ValueError(f"span invariant failed: {ent}")
        if ent["type"] not in VALID_TYPES and ent["type"] not in {"IGNORE","UNMAPPED"}: raise ValueError("invalid type")
        if any(a not in VALID_ASSERTIONS for a in ent.get("assertions", [])): raise ValueError("invalid assertion")
        for c in ent.get("candidates", []): validate_candidate(c)
        key=(s,e,ent["text"],ent["type"])
        if key in seen: raise ValueError("duplicate entity")
        seen.add(key)
