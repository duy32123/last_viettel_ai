from __future__ import annotations
from typing import Any


def finalize_predictions(text: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid=[]
    for row in rows:
        start=row.get('start'); end=row.get('end')
        if isinstance(start,int) and isinstance(end,int) and 0 <= start < end <= len(text) and text[start:end] == row.get('text'):
            valid.append(row)
    ranked=sorted(valid,key=lambda r:(-float(r.get('score',0.0)),-(r['end']-r['start']),r['start']))
    kept=[]
    for row in ranked:
        if not any(not (row['end'] <= old['start'] or row['start'] >= old['end']) for old in kept):
            kept.append(row)
    return sorted(kept,key=lambda r:(r['start'],r['end'],r.get('type','')))


def predict_with_model(text: str, tokenizer: Any, model: Any, *, max_length: int = 256, stride: int = 64) -> list[dict[str, Any]]:
    raise NotImplementedError('transformers NER inference is unavailable in this lightweight checkout')


def merge_chunk_predictions(chunks):
    best={}
    for chunk in chunks:
        for row in chunk:
            key=(row['start'],row['end']); cur=best.get(key)
            if cur is None or row.get('score',0.0) > cur.get('score',0.0): best[key]=dict(row)
    return [best[k] for k in sorted(best)]
