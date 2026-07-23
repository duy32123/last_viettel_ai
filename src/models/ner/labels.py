from __future__ import annotations
from src.data.dataset_schema import VALID_TYPES
TARGET_TYPES = sorted(VALID_TYPES)
LABELS = ['O'] + [f'{p}-{t}' for t in TARGET_TYPES for p in ('B','I','L','U')]
LABEL2ID = {l:i for i,l in enumerate(LABELS)}
ID2LABEL = {i:l for l,i in LABEL2ID.items()}

def labels_to_spans(offsets, labels, text_length=None):
    spans=[]; i=0; n=len(labels)
    while i<n:
        lab=labels[i]
        s,e=offsets[i]
        if lab == 'O' or s == e or e <= s or (text_length is not None and e > text_length):
            i+=1; continue
        if '-' not in lab: i+=1; continue
        pref,typ=lab.split('-',1)
        if pref=='U':
            spans.append({'start':s,'end':e,'type':typ}); i+=1; continue
        if pref=='B':
            start=s; j=i+1; end=None
            while j<n:
                lj=labels[j]; sj,ej=offsets[j]
                if sj==ej or ej<=sj: break
                if lj == f'L-{typ}': end=ej; break
                if lj != f'I-{typ}': break
                j+=1
            if end is not None:
                spans.append({'start':start,'end':end,'type':typ}); i=j+1; continue
        i+=1
    return [x for x in spans if x['start'] < x['end'] and (text_length is None or x['end'] <= text_length)]
