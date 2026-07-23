from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .labels import ASSERTION_LABELS, from_scores, ordered
from .rules import rule_assertions

def load_thresholds(path: str | Path) -> dict[str, float]:
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    return {l:float(data.get(l,{}).get('threshold', data.get(l,0.5)) if isinstance(data.get(l),dict) else data.get(l,0.5)) for l in ASSERTION_LABELS}

def merge_rule_model(rule_labels, scores, thresholds, *, text='', entity=None):
    labels=set(rule_labels or [])
    proposed=set(from_scores(scores, thresholds))
    if entity is not None:
        evidence=set(rule_assertions(text, entity)['labels'])
        proposed &= evidence
    labels |= proposed
    return ordered(labels)

def serialize_assertions(ent):
    return {k:ent[k] for k in ['text','type','position','candidates'] if k in ent} | {'assertions':ordered(ent.get('assertions',[]))}

def _model_scores(text, entities, model, tokenizer, batch_size=8):
    import torch
    return [[0.0,0.0,0.0] for _ in entities]

def predict_assertions(text: str, entities: list[dict[str, Any]], model: Any = None, tokenizer: Any = None, thresholds: dict[str,float] | None = None, model_scores_override=None) -> list[dict[str, Any]]:
    scores=model_scores_override if model_scores_override is not None else ([[0,0,0] for _ in entities] if model is None else _model_scores(text, entities, model, tokenizer))
    thresholds=thresholds or {l:0.5 for l in ASSERTION_LABELS}
    out=[]
    for ent,sc in zip(entities,scores):
        row=dict(ent); rule=rule_assertions(text, ent)['labels']
        row['assertions']=merge_rule_model(rule, sc, thresholds, text=text, entity=ent)
        out.append(row)
    return out
