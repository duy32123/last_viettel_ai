from __future__ import annotations
import json, warnings
from pathlib import Path
from typing import Any
from .labels import ordered, from_scores, ASSERTION_LABELS
from .rules import rule_assertions

def serialize_assertions(entity: dict[str,Any]) -> dict[str,Any]:
    return {"text":entity["text"],"type":entity["type"],"position":list(entity["position"]),"candidates":entity.get("candidates", []),"assertions":ordered(entity.get("assertions", []))}

def load_thresholds(path: str|Path|None) -> dict[str,float]:
    if not path or not Path(path).exists():
        warnings.warn("assertion thresholds.json not found; falling back to 0.5", RuntimeWarning)
        return {l:0.5 for l in ASSERTION_LABELS}
    raw=json.loads(Path(path).read_text(encoding="utf-8"))
    return {l:float(raw.get(l, {}).get("threshold", raw.get(l, 0.5))) for l in ASSERTION_LABELS}

def merge_rule_model(rule_labels: list[str], model_scores: list[float] | None=None, thresholds: dict[str,float] | None=None) -> list[str]:
    labels=set(rule_labels or [])
    if model_scores is not None: labels.update(from_scores(model_scores, thresholds))
    return ordered(labels)

def predict_assertions(text: str, entities: list[dict[str,Any]], model=None, tokenizer=None, thresholds: dict[str,float] | None=None, batch_size:int=16) -> list[dict[str,Any]]:
    out=[]
    for ent in entities:
        before=dict(ent)
        start,end=before["position"]
        if text[start:end] != before["text"]: raise ValueError("entity offset invariant failed")
        res=rule_assertions(text, ent)
        labels=res["labels"]
        # Model path is optional for unit/offline use; train/predict scripts supply scores in training environments.
        e={"text":before["text"],"type":before["type"],"position":list(before["position"]),"candidates":before.get("candidates", []),"assertions":merge_rule_model(labels, None, thresholds)}
        out.append(e)
    return out
