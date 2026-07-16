from __future__ import annotations
from typing import Any
from .labels import ordered, from_scores
from .rules import rule_assertions

def serialize_assertions(entity: dict[str,Any]) -> dict[str,Any]:
    return {"text":entity["text"],"type":entity["type"],"position":list(entity["position"]),"candidates":entity.get("candidates", []),"assertions":ordered(entity.get("assertions", []))}

def merge_rule_model(rule_labels: list[str], model_scores: list[float] | None=None, thresholds: dict[str,float] | None=None) -> list[str]:
    labels=set(rule_labels or [])
    if model_scores is not None: labels.update(from_scores(model_scores, thresholds))
    return ordered(labels)

def predict_assertions(text: str, entities: list[dict[str,Any]], model=None, tokenizer=None, thresholds: dict[str,float] | None=None) -> list[dict[str,Any]]:
    out=[]
    for ent in entities:
        before=dict(ent)
        res=rule_assertions(text, ent)
        labels=res["labels"]
        # Model path intentionally optional; full model inference belongs to training env.
        e={"text":before["text"],"type":before["type"],"position":list(before["position"]),"candidates":before.get("candidates", []),"assertions":merge_rule_model(labels, None, thresholds)}
        out.append(e)
    return out
