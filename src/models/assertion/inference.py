from __future__ import annotations
import json, re, unicodedata, warnings
from pathlib import Path
from typing import Any
from .labels import ordered, from_scores, ASSERTION_LABELS
from .preprocess import make_examples, register_special_tokens, tokenize_examples
from .rules import rule_assertions

def serialize_assertions(entity: dict[str,Any]) -> dict[str,Any]:
    return {"text":entity["text"],"type":entity["type"],"position":list(entity["position"]),"candidates":entity.get("candidates", []),"assertions":ordered(entity.get("assertions", []))}

def load_thresholds(path: str|Path|None) -> dict[str,float]:
    if not path or not Path(path).exists():
        return {l:0.5 for l in ASSERTION_LABELS}
    raw=json.loads(Path(path).read_text(encoding="utf-8"))
    out={}
    for lab in ASSERTION_LABELS:
        val=raw.get(lab, 0.5)
        out[lab]=float(val.get("threshold", 0.5) if isinstance(val, dict) else val)
    return out

_BREAK_RE=re.compile(r"(?:\bnhưng\b|\bnhung\b|\btuy nhiên\b|\btuy nhien\b|\bsong\b|\btuy vậy\b|[.;!?\n\r])", re.I)
_CURRENT_RE=re.compile(r"\b(?:hiện tại|hien tai|khám hiện tại|kham hien tai|lý do vào viện|ly do vao vien|vào viện|vao vien|đợt này|dot nay)\b", re.I)
_NEG_IDIOM_RE=re.compile(r"\b(?:không|khong)\s+những\b.+\b(?:mà|ma)\s+(?:còn|con)\b", re.I)
_CUES={
    "isFamily":["tiền sử gia đình","tien su gia dinh","gia đình","gia dinh","người nhà","nguoi nha","thân nhân","than nhan","mẹ","me","bố","bo","cha","anh ruột","anh ruot","chị ruột","chi ruot","em ruột","em ruot","ông","ong","bà","họ hàng","ho hang","dòng họ","dong ho"],
    "isHistorical":["tiền sử","tien su","trước đây","truoc day","đã từng","da tung","từng","tung","lần trước","lan truoc","hồ sơ cũ","ho so cu","nhiều năm trước","nhieu nam truoc","bệnh sử cũ","benh su cu","thông tin trước đây","thong tin truoc day"],
    "isNegated":["không ghi nhận","khong ghi nhan","không","khong","chưa","chua","chẳng","chang","phủ nhận","phu nhan","loại trừ","loai tru"],
}

def _fold(text: str) -> str:
    no_marks="".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", no_marks).casefold()

def _same_scope(text: str, cue_end: int, ent_start: int) -> bool:
    if cue_end > ent_start:
        return False
    between=text[cue_end:ent_start]
    if _BREAK_RE.search(between):
        return False
    return True

def _has_cue_before(text: str, start: int, cues: list[str], left: int=120) -> bool:
    lo=max(0, start-left)
    prefix=text[lo:start]
    prefix_fold=_fold(prefix)
    for cue in cues:
        cue_fold=_fold(cue)
        matches=list(re.finditer(r"(?<!\w)" + re.escape(cue_fold) + r"(?!\w)", prefix_fold))
        if matches:
            m=matches[-1]
            cue_end=lo+m.end()
            if _same_scope(_fold(text), cue_end, start):
                return True
    return False

def evidence_guard(label: str, text: str, entity: dict[str,Any]) -> bool:
    start,end=entity["position"] if "position" in entity else (entity["start"], entity["end"])
    if text[start:end] != entity["text"]:
        raise ValueError("entity offset invariant failed")
    folded=_fold(text)
    if label == "isNegated":
        lo,hi=max(0,start-120),min(len(text),end+80)
        if _NEG_IDIOM_RE.search(folded[lo:hi]):
            return False
        return _has_cue_before(text, start, _CUES[label], left=120)
    if label == "isFamily":
        lo=max(0,start-140)
        prefix=folded[lo:start]
        if re.search(r"(?:tien su\s+)?(?:benh nhan|nguoi benh)\s*$", prefix):
            return False
        return _has_cue_before(text, start, _CUES[label], left=140)
    if label == "isHistorical":
        lo=max(0,start-220)
        prefix=folded[lo:start]
        last_current=max((m.end() for m in _CURRENT_RE.finditer(prefix)), default=-1)
        search_start=lo + last_current if last_current >= 0 else lo
        return _has_cue_before(text[search_start:], start-search_start, _CUES[label], left=220)
    return False

def merge_rule_model(rule_labels: list[str], model_scores: list[float] | None=None, thresholds: dict[str,float] | None=None, text: str|None=None, entity: dict[str,Any]|None=None) -> list[str]:
    labels=set(rule_labels or [])
    if model_scores is not None:
        for lab in from_scores(model_scores, thresholds):
            if text is not None and entity is not None and not evidence_guard(lab, text, entity):
                continue
            labels.add(lab)
    return ordered(labels)

def _model_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"

def _model_scores(text: str, entities: list[dict[str,Any]], model, tokenizer, batch_size:int=16, max_length:int|None=None) -> list[list[float]]:
    import torch
    examples=make_examples([{"id":"doc","text":text,"entities":entities}])
    if not examples:
        return []
    register_special_tokens(tokenizer)
    max_len=max_length or int(getattr(getattr(model, "config", None), "max_position_embeddings", 256) or 256)
    max_len=min(max_len, int(getattr(tokenizer, "model_max_length", max_len) or max_len))
    scores=[]
    device=_model_device(model)
    model.eval()
    for i in range(0, len(examples), batch_size):
        batch_examples=examples[i:i+batch_size]
        enc=tokenize_examples(batch_examples, tokenizer, max_length=max_len, padding=True, truncation=True)
        inputs={k:torch.tensor(v, device=device) for k,v in enc.items() if k in {"input_ids","attention_mask","token_type_ids"}}
        with torch.no_grad():
            logits=model(**inputs).logits
            probs=torch.sigmoid(logits).detach().cpu().tolist()
        scores.extend(probs)
    return scores

def predict_assertions(text: str, entities: list[dict[str,Any]], model=None, tokenizer=None, thresholds: dict[str,float] | None=None, batch_size:int=16, model_scores_override: list[list[float]]|None=None) -> list[dict[str,Any]]:
    if (model is None) != (tokenizer is None):
        raise ValueError("model and tokenizer must be provided together")
    model_probs=model_scores_override if model_scores_override is not None else (_model_scores(text, entities, model, tokenizer, batch_size=batch_size) if model is not None else [None]*len(entities))
    out=[]
    for ent, probs in zip(entities, model_probs):
        before=dict(ent)
        start,end=before["position"]
        if text[start:end] != before["text"]: raise ValueError("entity offset invariant failed")
        res=rule_assertions(text, ent)
        labels=res["labels"]
        e={"text":before["text"],"type":before["type"],"position":list(before["position"]),"candidates":before.get("candidates", []),"assertions":merge_rule_model(labels, probs, thresholds, text=text, entity=before)}
        out.append(e)
    return out
