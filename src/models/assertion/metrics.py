from __future__ import annotations
from collections import Counter, defaultdict
from .labels import ASSERTION_LABELS, ordered

def multilabel_metrics(gold: list[list[str]], pred: list[list[str]]) -> dict:
    tp=Counter(); fp=Counter(); fn=Counter(); exact=0; none_ok=0; none_total=0; combo_ok=0; combos=Counter()
    for g,p in zip(gold,pred):
        gs=set(ordered(g)); ps=set(ordered(p)); combos["+".join(ordered(g)) or "NONE"] += 1
        exact += int(gs==ps); combo_ok += int(gs==ps)
        if not gs:
            none_total += 1; none_ok += int(not ps)
        for lab in ASSERTION_LABELS:
            tp[lab]+=int(lab in gs and lab in ps); fp[lab]+=int(lab not in gs and lab in ps); fn[lab]+=int(lab in gs and lab not in ps)
    per={}
    for lab in ASSERTION_LABELS:
        prec=tp[lab]/(tp[lab]+fp[lab]) if tp[lab]+fp[lab] else 0.0
        rec=tp[lab]/(tp[lab]+fn[lab]) if tp[lab]+fn[lab] else 0.0
        per[lab]={"precision":prec,"recall":rec,"f1":2*prec*rec/(prec+rec) if prec+rec else 0.0,"false_positive":fp[lab],"false_negative":fn[lab]}
    T=sum(tp.values()); Fp=sum(fp.values()); Fn=sum(fn.values())
    mp=T/(T+Fp) if T+Fp else 0.0; mr=T/(T+Fn) if T+Fn else 0.0
    return {"micro_precision":mp,"micro_recall":mr,"micro_f1":2*mp*mr/(mp+mr) if mp+mr else 0.0,"macro_f1":sum(v["f1"] for v in per.values())/len(ASSERTION_LABELS),"per_label":per,"subset_accuracy":exact/len(gold) if gold else 0.0,"none_accuracy":none_ok/none_total if none_total else None,"combination_accuracy":combo_ok/len(gold) if gold else 0.0,"combination_counts":dict(combos)}

def labels_from_scores(scores, thresholds):
    return [ordered([lab for lab,score in zip(ASSERTION_LABELS,row) if float(score) >= float(thresholds.get(lab,0.5))]) for row in scores]

def tune_thresholds(y_true, y_scores, min_precision: float=0.95):
    for i,lab in enumerate(ASSERTION_LABELS):
        vals=[row[i] for row in y_true]
        if not any(vals) or all(vals):
            raise ValueError(f"cannot tune threshold for {lab}: dev set must contain positive and negative examples")
    best={}
    for i,lab in enumerate(ASSERTION_LABELS):
        best_t=None; best_rec=-1.0; best_f=-1.0; best_prec=0.0
        for t in [x/100 for x in range(5,96,5)]:
            g=[[lab] if row[i] else [] for row in y_true]; p=[[lab] if row[i] >= t else [] for row in y_scores]
            vals=multilabel_metrics(g,p)["per_label"][lab]
            if vals["precision"] < min_precision:
                continue
            if vals["recall"] > best_rec or (vals["recall"] == best_rec and (vals["f1"] > best_f or (vals["f1"] == best_f and (best_t is None or t > best_t)))):
                best_t=t; best_rec=vals["recall"]; best_f=vals["f1"]; best_prec=vals["precision"]
        if best_t is None:
            best[lab]={"threshold":1.01,"disabled":True,"reason":f"no threshold reached precision >= {min_precision}","precision":0.0,"recall":0.0,"f1":0.0}
        else:
            best[lab]={"threshold":best_t,"disabled":False,"precision":best_prec,"recall":best_rec,"f1":best_f,"min_precision":min_precision}
    return best
