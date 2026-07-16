from __future__ import annotations
from typing import Any
from .preprocess import preprocess_records, decode_feature_spans


def merge_chunk_predictions(chunks: list[list[dict[str,Any]]]) -> list[dict[str,Any]]:
    best={}
    for spans in chunks:
        for s in spans:
            key=(s["start"], s["end"], s["type"])
            cur=best.get(key)
            if cur is None or s.get("score", 0) > cur.get("score", 0): best[key]=s
    return [best[k] for k in sorted(best)]


def predict_with_model(text: str, tokenizer, model, id2label: dict[int,str], max_length:int=256, stride:int=64) -> list[dict[str,Any]]:
    try:
        import torch
    except Exception as e:
        raise RuntimeError("torch is required for model inference") from e
    rec={"id":"doc","text":text,"entities":[]}
    features,_=preprocess_records([rec], tokenizer, max_length, stride)
    chunk_spans=[]
    for f in features:
        inputs={"input_ids":torch.tensor([f["input_ids"]]), "attention_mask":torch.tensor([f["attention_mask"]])}
        with torch.no_grad(): logits=model(**inputs).logits[0]
        ids=logits.argmax(dim=-1).tolist()
        spans=decode_feature_spans(f, ids)
        for s in spans:
            s["text"]=text[s["start"]:s["end"]]
            if text[s["start"]:s["end"]] != s["text"]: raise ValueError("prediction offset invariant failed")
        chunk_spans.append(spans)
    return merge_chunk_predictions(chunk_spans)
