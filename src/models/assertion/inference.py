from __future__ import annotations
import json, warnings
from pathlib import Path
from typing import Any
from .labels import ordered, from_scores, ASSERTION_LABELS
from .preprocess import make_examples, register_special_tokens, tokenize_examples
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

def predict_assertions(text: str, entities: list[dict[str,Any]], model=None, tokenizer=None, thresholds: dict[str,float] | None=None, batch_size:int=16) -> list[dict[str,Any]]:
    if (model is None) != (tokenizer is None):
        raise ValueError("model and tokenizer must be provided together")
    model_probs=_model_scores(text, entities, model, tokenizer, batch_size=batch_size) if model is not None else [None]*len(entities)
    out=[]
    for ent, probs in zip(entities, model_probs):
        before=dict(ent)
        start,end=before["position"]
        if text[start:end] != before["text"]: raise ValueError("entity offset invariant failed")
        res=rule_assertions(text, ent)
        labels=res["labels"]
        e={"text":before["text"],"type":before["type"],"position":list(before["position"]),"candidates":before.get("candidates", []),"assertions":merge_rule_model(labels, probs, thresholds)}
        out.append(e)
    return out
