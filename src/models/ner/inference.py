from __future__ import annotations
from typing import Any
from .preprocess import preprocess_records, decode_feature_spans


def merge_chunk_predictions(chunks: list[list[dict[str,Any]]]) -> list[dict[str,Any]]:
    best_by_boundary={}
    for spans in chunks:
        for s in spans:
            boundary=(s["start"], s["end"])
            cur=best_by_boundary.get(boundary)
            if cur is None or s.get("score", 0.0) > cur.get("score", 0.0):
                best_by_boundary[boundary]=s
    return [best_by_boundary[k] for k in sorted(best_by_boundary)]


def _model_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"


def _span_scores(offsets, pred_ids, token_scores, spans):
    for span in spans:
        idxs=[i for i,(s,e) in enumerate(offsets) if s!=e and s >= span["start"] and e <= span["end"]]
        if idxs:
            span["score"] = min(float(token_scores[i][pred_ids[i]]) for i in idxs)
        else:
            span["score"] = 0.0
    return spans


def predict_with_model(text: str, tokenizer, model, id2label: dict[int,str] | dict[str,str] | None=None, max_length:int=256, stride:int=64) -> list[dict[str,Any]]:
    try:
        import torch
    except Exception as e:
        raise RuntimeError("torch is required for model inference") from e
    model.eval()
    device=_model_device(model)
    config_id2label=id2label or getattr(model.config, "id2label", None)
    if not config_id2label: raise ValueError("model.config.id2label is required for NER inference")
    rec={"id":"doc","text":text,"entities":[]}
    features,_=preprocess_records([rec], tokenizer, max_length, stride)
    chunk_spans=[]
    for f in features:
        inputs={"input_ids":torch.tensor([f["input_ids"]], device=device), "attention_mask":torch.tensor([f["attention_mask"]], device=device)}
        with torch.no_grad():
            logits=model(**inputs).logits[0]
            probs=torch.softmax(logits, dim=-1)
        ids=logits.argmax(dim=-1).detach().cpu().tolist()
        scores=probs.detach().cpu().tolist()
        spans=decode_feature_spans(f, ids, config_id2label)
        spans=_span_scores(f["offset_mapping"], ids, scores, spans)
        for s in spans:
            s["text"]=text[s["start"]:s["end"]]
            if text[s["start"]:s["end"]] != s["text"]: raise ValueError("prediction offset invariant failed")
        chunk_spans.append(spans)
    return merge_chunk_predictions(chunk_spans)
