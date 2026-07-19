from __future__ import annotations
import argparse, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.kb_schema import read_jsonl
from src.linking.retrieval import LexicalIndex, load_lexical_index
from src.linking.dense import metrics_from_ranks

def _load(kb_dir):
    rows=[]
    for p in Path(kb_dir).glob('*.jsonl'): rows.extend(read_jsonl(p))
    return rows

def _examples(path):
    d=json.loads(Path(path).read_text())
    return d.get('splits', d)

def _candidate_dict(c):
    d=c.to_dict(); d['tty']=getattr(c,'tty',None); d['matched_alias']=getattr(c,'matched_alias',d['canonical_name']); return d

def _metrics(ranks): return metrics_from_ranks(ranks)

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/linking.rxnorm_bge_reranker.yaml'); p.add_argument('--pilot-examples', default='data/processed/linking_kb_rxnorm/rxnorm_pilot_examples.json'); p.add_argument('--mode', choices=['bm25','bge_dense','hybrid_rrf','reranker'], default='bm25'); p.add_argument('--max-eval-queries', type=int, default=0); p.add_argument('--dry-run', action='store_true'); ns=p.parse_args(argv)
    cfg=json.loads(Path(ns.config).read_text()); records=_load(cfg['kb_dir']); verified_codes={r.code for r in records if r.verified}
    if ns.dry_run:
        print(json.dumps({'records':len(records),'verified_rxcuis':len(verified_codes),'would_evaluate':True,'mode':ns.mode,'official_evaluation':False}, indent=2)); return
    if bool(cfg.get('production',False)) and any(not r.verified for r in records): raise ValueError('production RxNorm KB contains unverified records')
    splits=_examples(ns.pilot_examples); idx=LexicalIndex(records, include_unverified=bool(cfg.get('include_unverified',False)))
    out={'official_evaluation':False,'synthetic_pilot':True,'task':'rxnorm_medication_linking_pilot','mode':ns.mode,'candidate_universe':len(verified_codes),'splits':{},'per_tty':{},'per_task':{},'candidate_count_distribution':{},'unresolved_rate':0.0,'reranker_improvement':'not_run_offline'}; all_ranks=[]; tty_ranks=defaultdict(list); task_ranks=defaultdict(list); counts=Counter(); unresolved=0; total=0
    for sp,examples in splits.items():
        eval_examples=[e for e in examples if e.get('task')!='exact_alias_diagnostic']
        if ns.max_eval_queries: eval_examples=eval_examples[:ns.max_eval_queries]
        ranks=[]; start=time.time()
        for ex in eval_examples:
            query=ex.get('context') or ex.get('mention')
            cands=[_candidate_dict(c) for c in idx.search(query, 'THUỐC', top_k=int(cfg.get('top_k',20)), use_fuzzy=False)]
            counts[len(cands)] += 1; total += 1
            if not cands: unresolved += 1
            codes=[c['code'] for c in cands]; rank=codes.index(ex['positive_code'])+1 if ex['positive_code'] in codes else None
            ranks.append(rank); all_ranks.append(rank); tty_ranks[ex.get('tty','unknown')].append(rank); task_ranks[ex.get('task','unknown')].append(rank)
        m=_metrics(ranks); latency=time.time()-start; m.update({'query_count':len(eval_examples),'latency_sec':latency,'throughput_qps':len(eval_examples)/(latency or 1.0)}); out['splits'][sp]=m
    out['overall_main_excluding_exact_alias_diagnostic']=_metrics(all_ranks); out['per_tty']={k:_metrics(v) for k,v in sorted(tty_ranks.items())}; out['per_task']={k:_metrics(v) for k,v in sorted(task_ranks.items())}; out['candidate_count_distribution']={str(k):v for k,v in sorted(counts.items())}; out['unresolved_rate']=unresolved/(total or 1); out['readiness_basis']='held_out_synthetic_test_excluding_exact_alias_diagnostic_experimental'
    print(json.dumps(out, ensure_ascii=False, indent=2)); return out
if __name__=='__main__': main()
