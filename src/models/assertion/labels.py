from __future__ import annotations
ASSERTION_LABELS=["isNegated","isFamily","isHistorical"]
LABEL2ID={l:i for i,l in enumerate(ASSERTION_LABELS)}
ID2LABEL={i:l for l,i in LABEL2ID.items()}

def ordered(labels):
    s=set(labels or [])
    bad=s-set(ASSERTION_LABELS)
    if bad: raise ValueError(f"invalid assertion labels: {sorted(bad)}")
    return [l for l in ASSERTION_LABELS if l in s]

def to_vector(labels):
    s=set(ordered(labels)); return [1 if l in s else 0 for l in ASSERTION_LABELS]

def from_scores(scores, thresholds=None):
    thresholds=thresholds or {l:0.5 for l in ASSERTION_LABELS}
    return ordered([l for l,score in zip(ASSERTION_LABELS, scores) if float(score) >= float(thresholds.get(l,0.5))])
