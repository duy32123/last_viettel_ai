ASSERTION_LABELS=['isNegated','isFamily','isHistorical']
def ordered(labels): return [x for x in ASSERTION_LABELS if x in (labels or [])]
def to_vector(labels): return [1 if x in (labels or []) else 0 for x in ASSERTION_LABELS]
def from_scores(scores, thresholds): return [l for l,s in zip(ASSERTION_LABELS,scores) if float(s) >= float(thresholds.get(l,0.5))]

LABEL2ID={l:i for i,l in enumerate(ASSERTION_LABELS)}
ID2LABEL={i:l for l,i in LABEL2ID.items()}
