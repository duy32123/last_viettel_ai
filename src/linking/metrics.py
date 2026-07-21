from __future__ import annotations
from collections import defaultdict

def _gold_map(rows):
    m={}
    for r in rows:
        for e in r.get('entities',[]):
            gold=e.get('gold_code') or e.get('code') or (e.get('candidates') or [None])[0]
            if gold: m[(r.get('id'), tuple(e.get('position',[e.get('start'),e.get('end')])), e.get('type'))]=gold if isinstance(gold,str) else gold.get('code')
    return m

def evaluate(gold_rows, pred_rows):
    gold=_gold_map(gold_rows); ranks=[]; covered=0; abstain=0; buckets=defaultdict(lambda:{'n':0,'hit1':0})
    for r in pred_rows:
        for e in r.get('entities',[]):
            key=(r.get('id'), tuple(e.get('position',[e.get('start'),e.get('end')])), e.get('type'))
            if key not in gold: continue
            c=e.get('candidates',[]); codes=[x.get('code') if isinstance(x,dict) else x for x in c]
            if not c: abstain+=1; ranks.append(None); continue
            covered+=1; method=c[0].get('match_method','unknown') if isinstance(c[0],dict) else 'legacy'; buckets[method]['n']+=1
            rank=(codes.index(gold[key])+1) if gold[key] in codes else None; ranks.append(rank); buckets[method]['hit1']+=int(rank==1)
    n=len(ranks) or 1
    out={'recall@1':sum(1 for r in ranks if r and r<=1)/n,'recall@5':sum(1 for r in ranks if r and r<=5)/n,'recall@10':sum(1 for r in ranks if r and r<=10)/n,'mrr':sum((1/r) for r in ranks if r)/n,'accuracy@1':sum(1 for r in ranks if r==1)/n,'candidate_coverage':covered/n,'abstention_rate':abstain/n,'incorrect_confident_links':sum(1 for r in ranks if r is None),'bucket_metrics':{k:{'accuracy@1':v['hit1']/v['n'] if v['n'] else 0,'n':v['n']} for k,v in buckets.items()}}
    return out
