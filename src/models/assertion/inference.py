from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .labels import ASSERTION_LABELS, from_scores, ordered
from .preprocess import make_examples, tokenize_examples
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
    if not entities:
        return []
    records=[{'id':'inference','text':text,'entities':entities}]
    examples=make_examples(records)
    if hasattr(model, 'eval'):
        model.eval()
    scores=[]
    for start in range(0, len(examples), max(1, int(batch_size))):
        batch_examples=examples[start:start+max(1, int(batch_size))]
        encoded=tokenize_examples(batch_examples, tokenizer, padding=True)
        labels=encoded.pop('labels', None)
        inputs={}
        for key,value in encoded.items():
            if key == 'token_type_ids' or key in {'input_ids','attention_mask'}:
                inputs[key]=torch.as_tensor(value, dtype=torch.long)
        try:
            device=next(model.parameters()).device
            inputs={k:v.to(device) for k,v in inputs.items()}
        except Exception:
            pass
        with torch.no_grad():
            output=model(**inputs)
            logits=output.logits if hasattr(output, 'logits') else output[0]
            probs=torch.sigmoid(logits).detach().cpu().tolist()
        scores.extend([[float(x) for x in row[:len(ASSERTION_LABELS)]] for row in probs])
    return scores


def predict_assertions(text: str, entities: list[dict[str, Any]], model: Any = None, tokenizer: Any = None, thresholds: dict[str,float] | None = None, model_scores_override=None) -> list[dict[str, Any]]:
    scores=model_scores_override if model_scores_override is not None else ([[0,0,0] for _ in entities] if model is None else _model_scores(text, entities, model, tokenizer))
    thresholds=thresholds or {l:0.5 for l in ASSERTION_LABELS}
    out=[]
    for ent,sc in zip(entities,scores):
        row=dict(ent); rule=rule_assertions(text, ent)['labels']
        row['assertions']=merge_rule_model(rule, sc, thresholds, text=text, entity=ent)
        out.append(row)
    return out
