from __future__ import annotations
import re, unicodedata
from dataclasses import dataclass
from typing import Any
from .labels import ordered

NEGATED="isNegated"; FAMILY="isFamily"; HISTORICAL="isHistorical"
_BREAK_RE=re.compile(r"(?:\bnhưng\b|\bnhung\b|\btuy nhiên\b|\btuy nhien\b|\bsong\b|\btuy vậy\b|[.;!?\n\r])", re.I)
NEG_CUES=["không ghi nhận","khong ghi nhan","không có dấu hiệu","khong co dau hieu","âm tính với","am tinh voi","phủ nhận","phu nhan","loại trừ","loai tru","không","khong","chưa","chua","chẳng","chang"]
FAMILY_CUES=["tiền sử gia đình","tien su gia dinh","gia đình","gia dinh","người nhà","nguoi nha","thân nhân","than nhan","mẹ","me","bố","bo","cha","anh ruột","anh ruot","chị ruột","chi ruot","em ruột","em ruot","ông","ong","bà","họ hàng","ho hang","dòng họ","dong ho"]
HIST_CUES=["tiền sử","tien su","trước đây","truoc day","đã từng","da tung","từng","tung","lần trước","lan truoc","hồ sơ cũ","ho so cu","nhiều năm trước","nhieu nam truoc","bệnh sử cũ","benh su cu","thông tin trước đây","thong tin truoc day","hồi nhỏ","hoi nho","thuốc trước nhập viện","thuoc truoc nhap vien","bệnh cũ","benh cu","đã điều trị","da dieu tri"]
CURRENT_CUES=["hiện tại","hien tai","khám hiện tại","kham hien tai","lý do vào viện","ly do vao vien","vào viện","vao vien","đợt này","dot nay"]

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

def _fold(text: str) -> str:
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")).casefold()

def _find_cue(text: str, cues: list[str], start: int, end: int, left=80):
    lo=max(0,start-left); prefix=_fold(text[lo:start])
    best=None
    for cue in cues:
        pattern=r"(?<!\w)" + re.escape(_fold(cue)) + r"(?!\w)"
        for m in re.finditer(pattern, prefix):
            abs_s=lo+m.start(); abs_e=lo+m.end()
            if best is None or abs_e > best[2]: best=(cue,abs_s,abs_e)
    return best

def _in_section(text: str, start: int, cue: str) -> bool:
    lo=max(0,start-200); segment=_fold(text[lo:start]); cue=_fold(cue)
    last_current=max((segment.rfind(_fold(cur)) for cur in CURRENT_CUES), default=-1)
    cpos=segment.rfind(cue)
    if cpos < 0: return False
    if last_current > cpos: return False
    after=segment[cpos:]
    return not any(_fold(cur) in after for cur in CURRENT_CUES)

def rule_assertions(text: str, entity: dict[str,Any]) -> dict[str,Any]:
    start,end=entity["position"] if "position" in entity else (entity["start"], entity["end"])
    if text[start:end] != entity["text"]: raise ValueError("entity offset invariant failed")
    hits=[]; low=_fold(text)
    # Negation: ignore idiom "không những ... mà còn".
    lo,hi=_window(text,start,end)
    local=low[lo:hi]
    if "khong nhung" not in local:
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
