from __future__ import annotations
from .labels import ASSERTION_LABELS, from_scores


def _prf(tp:int, fp:int, fn:int) -> dict[str,float|int]:
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':f1}


def multilabel_metrics(gold,pred):
    total_tp=total_fp=total_fn=0; per={}
    for label in ASSERTION_LABELS:
        tp=fp=fn=0
        for g,p in zip(gold,pred):
            gs=set(g); ps=set(p)
            tp += label in gs and label in ps
            fp += label not in gs and label in ps
            fn += label in gs and label not in ps
        vals=_prf(tp,fp,fn); vals['false_negative']=fn; per[label]=vals
        total_tp += tp; total_fp += fp; total_fn += fn
    micro=_prf(total_tp,total_fp,total_fn)
    return {'micro_precision':micro['precision'],'micro_recall':micro['recall'],'micro_f1':micro['f1'],'per_label':per}


def labels_from_scores(scores, thresholds):
    return from_scores(scores, thresholds)


def _label_f1(y_true, y_score, label_index:int, threshold:float):
    tp=fp=fn=0
    for true_row, score_row in zip(y_true,y_score):
        pred=float(score_row[label_index]) >= threshold
        gold=bool(true_row[label_index])
        tp += pred and gold; fp += pred and not gold; fn += (not pred) and gold
    return _prf(tp,fp,fn)


def tune_thresholds(y_true,y_score):
    if len(y_true) < 3 or len(y_true) != len(y_score):
        raise ValueError('threshold tuning requires matching dev labels and scores with at least 3 examples')
    width=len(ASSERTION_LABELS)
    if any(len(row) != width for row in y_true) or any(len(row) != width for row in y_score):
        raise ValueError('threshold tuning rows must match assertion label count')
    out={}
    for idx,label in enumerate(ASSERTION_LABELS):
        candidates=sorted({0.0,0.5,1.0, *[max(0.0,min(1.0,float(row[idx]))) for row in y_score]})
        # Midpoints avoid depending on exact equality while preserving useful cutoffs.
        mids={(a+b)/2 for a,b in zip(candidates, candidates[1:])}
        search=sorted(set(candidates)|mids)
        best=None
        for th in search:
            vals=_label_f1(y_true,y_score,idx,th)
            key=(vals['f1'], vals['precision'], vals['recall'], -abs(th-0.5))
            if best is None or key > best[0]:
                best=(key, th, vals)
        _, threshold, vals=best
        out[label]={'threshold':float(threshold), 'precision':vals['precision'], 'recall':vals['recall'], 'f1':vals['f1'], 'tp':vals['tp'], 'fp':vals['fp'], 'fn':vals['fn']}
    return out
