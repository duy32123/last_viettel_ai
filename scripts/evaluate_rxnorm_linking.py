from __future__ import annotations
import argparse, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.kb_schema import read_jsonl
from src.linking.retrieval import LexicalIndex
from src.linking.dense import metrics_from_ranks

def _load(kb_dir):
    rows=[]
    for p in Path(kb_dir).glob('*.jsonl'): rows.extend(read_jsonl(p))
    return rows

def _examples(path): return json.loads(Path(path).read_text())

def _candidate_dict(c):
    d=c.to_dict(); d['tty']=getattr(c,'tty',None); d['matched_alias']=getattr(c,'matched_alias',d['canonical_name']); return d

def _metrics(ranks): return metrics_from_ranks(ranks)

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/linking.rxnorm_bge_reranker.yaml'); p.add_argument('--pilot-examples', default='data/processed/linking_kb_rxnorm/rxnorm_pilot_examples.json'); p.add_argument('--dry-run', action='store_true'); ns=p.parse_args(argv)
    cfg=json.loads(Path(ns.config).read_text()); records=_load(cfg['kb_dir'])
    if ns.dry_run:
        print(json.dumps({'records':len(records),'would_evaluate':True,'official_evaluation':False}, indent=2)); return
    splits=_examples(ns.pilot_examples); idx=LexicalIndex(records, include_unverified=bool(cfg.get('include_unverified',False)))
    out={'official_evaluation':False,'task':'rxnorm_medication_linking_pilot','splits':{},'per_tty':{},'candidate_count_distribution':{},'unresolved_rate':0.0,'reranker_improvement':'not_run_offline'}; all_ranks=[]; tty_ranks=defaultdict(list); counts=Counter(); unresolved=0; total=0
    for sp,examples in splits.items():
        ranks=[]; start=time.time()
        for ex in examples:
            query=ex.get('context') or ex.get('mention')
            cands=[_candidate_dict(c) for c in idx.search(query, 'THUỐC', top_k=int(cfg.get('top_k',20)))]
            counts[len(cands)] += 1; total += 1
            if not cands: unresolved += 1
            codes=[c['code'] for c in cands]; rank=codes.index(ex['positive_code'])+1 if ex['positive_code'] in codes else None
            ranks.append(rank); all_ranks.append(rank); tty_ranks[ex.get('tty','unknown')].append(rank)
        m=_metrics(ranks); latency=time.time()-start; m.update({'query_count':len(examples),'latency_sec':latency,'throughput_qps':len(examples)/(latency or 1.0)}); out['splits'][sp]=m
    out['overall']=_metrics(all_ranks); out['per_tty']={k:_metrics(v) for k,v in sorted(tty_ranks.items())}; out['candidate_count_distribution']={str(k):v for k,v in sorted(counts.items())}; out['unresolved_rate']=unresolved/(total or 1)
    print(json.dumps(out, ensure_ascii=False, indent=2)); return out
if __name__=='__main__': main()
