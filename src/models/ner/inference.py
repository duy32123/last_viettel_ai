from __future__ import annotations
import re
from typing import Any
from src.models.ner.preprocess import normalize_predicted_span

LAB_NAMES={x.casefold() for x in ['HbA1c','HBA1C','CRP','INR','glucose','WBC','creatinine','AST','ALT','hemoglobin','Hb','ferritin','TSH','LDH','procalcitonin','D-dimer','albumin','natri','kali','ESR','troponin','Na+','SARS-CoV-2','cúm A']}
LAB_CUE_RE=re.compile(r'\b(xét nghiệm|kết quả|XN|lab|glucose|HbA1c|HBA1C|CRP|INR|WBC|creatinine|troponin|D-dimer|Na\+|AST|ALT|TSH|LDH|ESR|ferritin|albumin|kali|natri|procalcitonin)\b', re.I)
NON_LAB_RE=re.compile(r'\b(cao|nặng|uống thuốc|thuốc|giá|giảm|tuổi)\b', re.I)

def _flatten(rows):
    out=[]
    for r in rows or []:
        if isinstance(r, list): out.extend(_flatten(r))
        else: out.append(r)
    return out

def merge_chunk_predictions(chunks):
    best={}
    for row in _flatten(chunks):
        key=(row['start'],row['end']); cur=best.get(key)
        if cur is None or row.get('score',0.0) > cur.get('score',0.0): best[key]=dict(row)
    return [best[k] for k in sorted(best)]

def _has_lab_evidence(text: str, ent: dict[str,Any], all_rows: list[dict[str,Any]]) -> bool:
    s,e=ent['start'],ent['end']; window=text[max(0,s-50):min(len(text),e+50)]
    if any(r.get('type') == 'TÊN_XÉT_NGHIỆM' and abs(r.get('start',0)-s) <= 80 for r in all_rows): return True
    if LAB_CUE_RE.search(window): return True
    if NON_LAB_RE.search(window) and not LAB_CUE_RE.search(window): return False
    return False

def filter_non_lab_result_predictions(text: str, spans: list[dict[str,Any]]) -> list[dict[str,Any]]:
    rows=[]
    for row in _flatten(spans):
        if row.get('type') != 'KẾT_QUẢ_XÉT_NGHIỆM' or _has_lab_evidence(text,row,_flatten(spans)):
            rows.append(row)
    return rows

def finalize_predictions(text: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw=[]
    for row in _flatten(rows):
        start=row.get('start'); end=row.get('end')
        if not isinstance(start,int) or not isinstance(end,int): continue
        norm=normalize_predicted_span(text,start,end)
        if not norm: continue
        r=dict(row); r.update(norm)
        if text[r['start']:r['end']] != r.get('text'): continue
        raw.append(r)
    raw=filter_non_lab_result_predictions(text, raw)
    valid=[]
    for row in raw:
        # Keep lab names, non-lab concepts, and lab values with evidence; drop ordinary measurements.
        valid.append(row)
    ranked=sorted(valid,key=lambda r:(-float(r.get('score',0.0)),-(r['end']-r['start']),r['start']))
    kept=[]
    for row in ranked:
        if not any(not (row['end'] <= old['start'] or row['start'] >= old['end']) for old in kept): kept.append(row)
    return sorted(kept,key=lambda r:(r['start'],r['end'],r.get('type','')))

def predict_with_model(text: str, tokenizer: Any, model: Any, *, max_length: int = 256, stride: int = 64) -> list[dict[str, Any]]:
    from src.models.ner.preprocess import decode_feature_spans
    from src.models.ner.labels import ID2LABEL
    import torch
    enc=tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length, stride=stride, return_overflowing_tokens=True, padding=False)
    offsets_obj=enc['offset_mapping']
    batched=bool(offsets_obj and isinstance(offsets_obj[0], (list, tuple)) and offsets_obj[0] and isinstance(offsets_obj[0][0], (list, tuple)))
    count=len(offsets_obj) if batched else 1
    chunks=[]
    for i in range(count):
        input_ids=enc['input_ids'][i] if batched else enc['input_ids']
        attention=enc.get('attention_mask', [[1]*len(input_ids)])[i] if batched and 'attention_mask' in enc else enc.get('attention_mask', [1]*len(input_ids))
        offsets=enc['offset_mapping'][i] if batched else enc['offset_mapping']
        device=next(model.parameters()).device
        with torch.no_grad():
            out=model(input_ids=torch.tensor([input_ids], device=device), attention_mask=torch.tensor([attention], device=device))
        probs=torch.softmax(out.logits, dim=-1)[0].detach().cpu()
        pred=probs.argmax(dim=-1).tolist()
        token_conf=probs.max(dim=-1).values.tolist()
        spans=decode_feature_spans({'text':text,'offset_mapping':offsets}, pred, ID2LABEL)
        scored=[]
        for sp in spans:
            idxs=[j for j,(a,b) in enumerate(offsets) if a != b and max(a,sp['start']) < min(b,sp['end'])]
            score=sum(token_conf[j] for j in idxs)/len(idxs) if idxs else 0.0
            scored.append({**sp,'score':float(score)})
        chunks.append(scored)
    return finalize_predictions(text, chunks)
