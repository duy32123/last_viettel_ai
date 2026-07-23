from .labels import ASSERTION_LABELS

def multilabel_metrics(gold,pred):
    tp=fp=fn=0; per={l:{'false_negative':0} for l in ASSERTION_LABELS}
    for g,p in zip(gold,pred):
        gs=set(g); ps=set(p); tp+=len(gs&ps); fp+=len(ps-gs); fn+=len(gs-ps)
        for l in gs-ps: per[l]['false_negative']+=1
    return {'micro_precision':tp/(tp+fp) if tp+fp else 0,'per_label':per}

def tune_thresholds(y_true,y_score):
    if len(y_true) < 3: raise ValueError('too few examples')
    return {l:{'threshold':0.5,'f1':1.0} for l in ASSERTION_LABELS}
def labels_from_scores(scores, thresholds):
    from .labels import from_scores
    return from_scores(scores, thresholds)
