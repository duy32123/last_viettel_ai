from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Any
from .labels import ordered

NEGATED="isNegated"; FAMILY="isFamily"; HISTORICAL="isHistorical"
_BREAK_RE=re.compile(r"(?:\bnhưng\b|\btuy nhiên\b|\bsong\b|\btuy vậy\b|[.;!?\n\r])", re.I)
NEG_CUES=["không ghi nhận","không có dấu hiệu","âm tính với","phủ nhận","loại trừ","không","chưa","chẳng"]
FAMILY_CUES=["tiền sử gia đình","gia đình","người nhà","mẹ","bố","cha","ba","anh","chị","em","con","ông","bà"]
HIST_CUES=["tiền sử","trước đây","đã từng","từng","hồi nhỏ","thuốc trước nhập viện","bệnh cũ","đã điều trị"]
CURRENT_CUES=["hiện tại","khám hiện tại","lý do vào viện","vào viện"]

@dataclass
class RuleHit:
    label: str
    cue: str
    start: int
    end: int
    rule_id: str
    confidence: float=0.99
    provenance: str="rule"
    def asdict(self): return dict(label=self.label, cue=self.cue, cue_span=[self.start,self.end], rule_id=self.rule_id, confidence=self.confidence, provenance=self.provenance)

def _window(text, start, end, left=80, right=40):
    return max(0,start-left), min(len(text), end+right)

def _same_scope(text: str, cue_end: int, ent_start: int) -> bool:
    if cue_end > ent_start: return False
    between=text[cue_end:ent_start]
    return _BREAK_RE.search(between) is None

def _find_cue(text: str, cues: list[str], start: int, end: int, left=80):
    lo=max(0,start-left); prefix=text[lo:start].casefold()
    best=None
    for cue in cues:
        pattern=r"(?<!\w)" + re.escape(cue.casefold()) + r"(?!\w)"
        for m in re.finditer(pattern, prefix):
            abs_s=lo+m.start(); abs_e=lo+m.end()
            if best is None or abs_e > best[2]: best=(cue,abs_s,abs_e)
    return best

def _in_section(text: str, start: int, cue: str) -> bool:
    lo=max(0,start-200); segment=text[lo:start].casefold()
    cpos=segment.rfind(cue)
    if cpos < 0: return False
    after=segment[cpos:]
    return not any(cur in after for cur in CURRENT_CUES)

def rule_assertions(text: str, entity: dict[str,Any]) -> dict[str,Any]:
    start,end=entity["position"] if "position" in entity else (entity["start"], entity["end"])
    if text[start:end] != entity["text"]: raise ValueError("entity offset invariant failed")
    hits=[]; low=text.casefold()
    # Negation: ignore idiom "không những ... mà còn".
    lo,hi=_window(text,start,end)
    local=low[lo:hi]
    if "không những" not in local:
        cue=_find_cue(text, NEG_CUES, start, end)
        if cue and _same_scope(text, cue[2], start): hits.append(RuleHit(NEGATED, cue[0], cue[1], cue[2], "neg_scope").asdict())
    # Family cue before entity in same clause/section.
    cue=_find_cue(text, FAMILY_CUES, start, end, left=100)
    if cue and (_same_scope(text, cue[2], start) or cue[0] in {"tiền sử gia đình","gia đình","người nhà"}): hits.append(RuleHit(FAMILY, cue[0], cue[1], cue[2], "family_scope").asdict())
    # Historical cue section-aware.
    cue=_find_cue(text, HIST_CUES, start, end, left=160)
    if cue and (_same_scope(text, cue[2], start) or _in_section(text, start, cue[0])):
        hits.append(RuleHit(HISTORICAL, cue[0], cue[1], cue[2], "historical_scope").asdict())
    labels=ordered([h["label"] for h in hits])
    return {"labels":labels, "rule_hits":hits}

def apply_rules_to_entities(text: str, entities: list[dict[str,Any]]) -> list[dict[str,Any]]:
    out=[]
    for ent in entities:
        e=dict(ent)
        res=rule_assertions(text, e)
        e["assertions"]=res["labels"]
        e.setdefault("metadata", {})["assertion_rule_hits"]=res["rule_hits"]
        out.append(e)
    return out
