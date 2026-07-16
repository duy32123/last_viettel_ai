from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any
from .labels import TARGET_TYPES

def _key(e): return (e["start"], e["end"], e["type"])
def _boundary(e): return (e["start"], e["end"])

def evaluate_spans(gold_docs: list[dict[str,Any]], pred_docs: list[dict[str,Any]]) -> dict[str,Any]:
    pred_by_id={d["id"]:d for d in pred_docs}; tp=fp=fn=0; per={t:Counter() for t in TARGET_TYPES}; boundary_tp=boundary_fp=boundary_fn=0; type_ok=type_total=0; confusion=defaultdict(Counter); start_end_off=0; errors=[]
    for gd in gold_docs:
        pd=pred_by_id.get(gd["id"], {"entities":[]})
        gold=[e for e in gd.get("entities",[]) if e["type"] in TARGET_TYPES]
        pred=[e for e in pd.get("entities",[]) if e["type"] in TARGET_TYPES]
        gs={_key(e):e for e in gold}; ps={_key(e):e for e in pred}
        gkeys=set(gs); pkeys=set(ps); inter=gkeys&pkeys
        tp += len(inter); fp += len(pkeys-inter); fn += len(gkeys-inter)
        gb=defaultdict(list); pb=defaultdict(list)
        for e in gold: gb[_boundary(e)].append(e)
        for e in pred: pb[_boundary(e)].append(e)
        binter=set(gb)&set(pb); boundary_tp += len(binter); boundary_fp += len(set(pb)-set(gb)); boundary_fn += len(set(gb)-set(pb))
        for b in binter:
            type_total += 1
            gt=gb[b][0]["type"]; pt=pb[b][0]["type"]; confusion[gt][pt]+=1
            if gt == pt: type_ok += 1
        for e in pred:
            if not any(e["start"] == g["start"] or e["end"] == g["end"] for g in gold) and _key(e) not in gkeys:
                start_end_off += 1
        for t in TARGET_TYPES:
            g_t={_key(e) for e in gold if e["type"]==t}; p_t={_key(e) for e in pred if e["type"]==t}
            per[t]["tp"] += len(g_t&p_t); per[t]["fp"] += len(p_t-g_t); per[t]["fn"] += len(g_t-p_t)
        fps=[ps[k] for k in pkeys-gkeys]; fns=[gs[k] for k in gkeys-pkeys]
        errors.append({"id":gd["id"],"text":gd["text"],"gold_entities":gold,"predicted_entities":pred,"false_positives":fps,"false_negatives":fns,"boundary_errors":[p for p in fps if any(_boundary(p)==_boundary(g) for g in gold)],"type_errors":[p for p in fps if any(_boundary(p)==_boundary(g) and p["type"]!=g["type"] for g in gold)]})
    def prf(c):
        p=c["tp"]/(c["tp"]+c["fp"]) if c["tp"]+c["fp"] else 0.0; r=c["tp"]/(c["tp"]+c["fn"]) if c["tp"]+c["fn"] else 0.0; f=2*p*r/(p+r) if p+r else 0.0
        return {"precision":p,"recall":r,"f1":f,"tp":c["tp"],"fp":c["fp"],"fn":c["fn"]}
    overall=prf(Counter({"tp":tp,"fp":fp,"fn":fn})); boundary=prf(Counter({"tp":boundary_tp,"fp":boundary_fp,"fn":boundary_fn}))
    return {"strict_micro":overall,"per_type":{t:prf(per[t]) for t in TARGET_TYPES},"boundary_only":boundary,"type_accuracy_on_correct_boundary":type_ok/type_total if type_total else 0.0,"start_end_offset_errors":start_end_off,"false_positives":fp,"false_negatives":fn,"confusion_matrix":{g:dict(p) for g,p in confusion.items()},"errors":errors}
