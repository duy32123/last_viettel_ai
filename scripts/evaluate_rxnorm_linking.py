from __future__ import annotations
import argparse, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.kb_schema import read_jsonl
from src.linking.retrieval import LexicalIndex, load_lexical_index, kb_paths_checksum
from src.linking.dense import BGEM3Backend, BGERerankerBackend, DenseAliasIndex, MockDenseEncoder, MockReranker, dense_expected_manifest, load_dense_index, metrics_from_ranks, rrf_fuse, tune_rrf, rerank_candidates, tune_blend
from scripts.evaluate_hybrid_linking import _encode_query_vectors

def _paths(kb_dir): return sorted(Path(kb_dir).glob('*.jsonl'))
def _load(kb_dir):
    rows=[]
    for p in _paths(kb_dir): rows.extend(read_jsonl(p))
    return rows

def _examples(path):
    d=json.loads(Path(path).read_text())
    return d.get('splits', d)

def _candidate_dict(c):
    d=c.to_dict(); d['tty']=getattr(c,'tty',None); d['matched_alias']=getattr(c,'matched_alias',d['canonical_name']); return d

def _metrics(ranks): return metrics_from_ranks(ranks)

def _main_examples(splits, max_per_split=0):
    out={}
    for sp,examples in splits.items():
        rows=[e for e in examples if e.get('task')!='exact_alias_diagnostic']
        out[sp]=rows[:max_per_split] if max_per_split else rows
    return out

def _gate_split_leakage(splits):
    seen={}; leaks=[]
    for sp,examples in splits.items():
        for e in examples:
            code=e['positive_code']
            if code in seen and seen[code]!=sp: leaks.append((code,seen[code],sp))
            seen[code]=sp
    if leaks: raise ValueError(f'RxCUI split leakage detected: {leaks[:3]}')

def _canonical_overlap(records, splits):
    names={r.code:r.canonical_name.casefold() for r in records}; total=hit=0
    for examples in splits.values():
        for e in examples:
            if e.get('task')=='exact_alias_diagnostic': continue
            total += 1; name=names.get(e['positive_code'],'').casefold(); query=(e.get('context') or e.get('mention') or '').casefold()
            if name and name in query: hit += 1
    return {'query_contains_canonical_name':hit>0,'canonical_name_overlap_rate':hit/(total or 1)}

def _eval_code_ranker(splits, ranker):
    out={'splits':{},'per_tty':{},'per_task':{},'candidate_count_distribution':{}}; all_ranks=[]; tty=defaultdict(list); task=defaultdict(list); counts=Counter(); unresolved=0; total=0
    for sp,examples in splits.items():
        ranks=[]; start=time.time()
        for e in examples:
            codes=ranker(e); counts[len(codes)] += 1; total += 1
            if not codes: unresolved += 1
            rank=codes.index(e['positive_code'])+1 if e['positive_code'] in codes else None
            ranks.append(rank); all_ranks.append(rank); tty[e.get('tty','unknown')].append(rank); task[e.get('task','unknown')].append(rank)
        m=_metrics(ranks); lat=time.time()-start; m.update({'query_count':len(examples),'latency_sec':lat,'throughput_qps':len(examples)/(lat or 1.0)}); out['splits'][sp]=m
    out['overall_main_excluding_exact_alias_diagnostic']=_metrics(all_ranks); out['per_tty']={k:_metrics(v) for k,v in sorted(tty.items())}; out['per_task']={k:_metrics(v) for k,v in sorted(task.items())}; out['candidate_count_distribution']={str(k):v for k,v in sorted(counts.items())}; out['unresolved_rate']=unresolved/(total or 1)
    return out

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/linking.rxnorm_bge_reranker.yaml'); p.add_argument('--pilot-examples', default='data/processed/linking_kb_rxnorm/rxnorm_pilot_examples.json'); p.add_argument('--mode', choices=['bm25','bge_dense','hybrid_rrf','reranker'], default='bm25'); p.add_argument('--max-eval-queries-per-split', type=int, default=0); p.add_argument('--max-eval-queries', type=int, default=0, help='Deprecated global cap alias; treated as per-split for backward compatibility.'); p.add_argument('--mock-dense', action='store_true'); p.add_argument('--mock-reranker', action='store_true'); p.add_argument('--dry-run', action='store_true'); ns=p.parse_args(argv)
    cfg=json.loads(Path(ns.config).read_text()); kb_dir=Path(cfg['kb_dir']); records=_load(kb_dir); paths=_paths(kb_dir); verified_codes={r.code for r in records if r.verified}; requested=ns.mode
    if ns.dry_run:
        print(json.dumps({'requested_mode':requested,'executed_backend':None,'dry_run':True,'backend_executed':False,'records':len(records),'verified_rxcuis':len(verified_codes),'expected_artifacts':{'lexical_index_dir':cfg.get('index_dir'),'dense_index_dir':cfg.get('dense_index_dir')},'official_evaluation':False}, indent=2)); return
    if bool(cfg.get('production',False)) and any(not r.verified for r in records): raise ValueError('production RxNorm KB contains unverified records')
    raw_splits=_examples(ns.pilot_examples); _gate_split_leakage(raw_splits); cap=ns.max_eval_queries_per_split or ns.max_eval_queries; splits=_main_examples(raw_splits, cap)
    include_unverified=bool(cfg.get('include_unverified',False)); top_k=int(cfg.get('top_k',20)); retrieval_depth=int(cfg.get('retrieval_depth',100)); expected_checksum=kb_paths_checksum(paths); candidate_universe=len(verified_codes); counters={'dense_encode_calls':0,'dense_search_calls':0,'bm25_search_calls':0,'ranking_cache_hits':0,'reranker_pairs_scored':0,'reranker_batches':0}
    bm25_called=dense_called=rerank_called=False; model_name=model_revision=device=None; cache_validation=None; selected_on_dev=None; selected_blend_on_dev=None; fallback_used=False; dev_before=None; dev_after=None
    lexical=None
    def _key(e): return e.get('id') or f"{e.get('positive_code')}:{e.get('task')}:{e.get('context') or e.get('mention')}"
    def bm25_rank(e):
        nonlocal bm25_called; bm25_called=True
        key=('bm25',_key(e))
        if key in rank_cache: counters['ranking_cache_hits'] += 1; return rank_cache[key]
        counters['bm25_search_calls'] += 1; rank_cache[key]=[c.code for c in lexical.search(e.get('context') or e.get('mention'), 'THUỐC', top_k=retrieval_depth, use_fuzzy=False)]; return rank_cache[key]
    dense_index=encoder=None; qvec_cache={}; rank_cache={}
    def load_dense():
        nonlocal dense_index,encoder,model_name,model_revision,device,cache_validation
        if dense_index is not None: return
        if ns.mock_dense:
            encoder=MockDenseEncoder(); dense_index=DenseAliasIndex.build(records, encoder, include_unverified=include_unverified); cache_validation='mock'; model_name='mock'; model_revision='mock'; device='mock'; return
        model_name=cfg.get('model_name','BAAI/bge-m3'); model_revision=cfg.get('model_revision','main')
        expected=dense_expected_manifest(cfg, paths, candidate_universe=candidate_universe)
        dense_index=load_dense_index(Path(cfg['dense_index_dir']), expected); encoder=BGEM3Backend(model_name=model_name,batch_size=int(cfg.get('batch_size',16)),max_length=int(cfg.get('max_length',8192)),use_fp16=cfg.get('use_fp16'),device=cfg.get('device')); device=getattr(encoder,'device',None) or 'auto'; cache_validation='valid'
    def ensure_qvecs():
        load_dense()
        if qvec_cache: return
        for sp,examples in splits.items():
            texts=[e.get('context') or e.get('mention') for e in examples]
            counters['dense_encode_calls'] += 1; vecs,_=_encode_query_vectors(encoder,texts,batch_size=int(cfg.get('batch_size',16)),expected_dim=dense_index._dim())
            for e,v in zip(examples,vecs): qvec_cache[id(e)]=v
    def dense_rank(e):
        nonlocal dense_called; dense_called=True; ensure_qvecs()
        key=('dense',_key(e))
        if key in rank_cache: counters['ranking_cache_hits'] += 1; return rank_cache[key]
        counters['dense_search_calls'] += 1; rank_cache[key]=[r['code'] for r in dense_index.search_vector(qvec_cache[id(e)], top_k=retrieval_depth)]; return rank_cache[key]
    lexical=load_lexical_index(kb_dir, include_unverified=include_unverified, expected_kb_checksum=expected_checksum, index_dir=Path(cfg.get('index_dir',kb_dir)))
    if requested=='bm25':
        executed='bm25'; ranker=bm25_rank
    elif requested=='bge_dense':
        executed='bge_dense'; load_dense(); ensure_qvecs(); ranker=dense_rank
    elif requested=='hybrid_rrf':
        executed='hybrid_rrf'; load_dense(); ensure_qvecs(); params=tune_rrf(splits.get('dev',[]), bm25_rank, dense_rank); selected_on_dev=params
        ranker=lambda e: rrf_fuse([(params['bm25_weight'],bm25_rank(e)),(params['dense_weight'],dense_rank(e))], params['rrf_k'], top_k=retrieval_depth)
    else:
        executed='reranker'; load_dense(); ensure_qvecs(); params=tune_rrf(splits.get('dev',[]), bm25_rank, dense_rank); selected_on_dev=params
        base_rank=lambda e: rrf_fuse([(params['bm25_weight'],bm25_rank(e)),(params['dense_weight'],dense_rank(e))], params['rrf_k'], top_k=retrieval_depth)
        reranker=MockReranker() if ns.mock_reranker else BGERerankerBackend(cfg.get('reranker_model_name','BAAI/bge-reranker-v2-m3'), int(cfg.get('reranker_batch_size',16)), int(cfg.get('reranker_max_length',8192)), cfg.get('use_fp16'), cfg.get('device'))
        by_code={r.code:r for r in records}
        def rows_for(e):
            rows=[]
            for i,code in enumerate(base_rank(e)[:20],1):
                rec=by_code[code]; rows.append({'code':code,'canonical_name':rec.canonical_name,'matched_alias':rec.canonical_name,'score':1.0/i,'rank':i,'verified':rec.verified,'source':rec.source,'version':rec.version})
            return rows
        def rerank_codes(e):
            nonlocal rerank_called; rerank_called=True
            rows=rows_for(e); counters['reranker_pairs_scored'] += len(rows); counters['reranker_batches'] += 1; out=rerank_candidates(e.get('context') or e.get('mention'), rows, reranker, batch_size=int(cfg.get('reranker_batch_size',16)))
            if {r['code'] for r in rows}!={r['code'] for r in out}: raise ValueError('reranker changed candidate membership')
            return [r['code'] for r in out]
        dev_base=_eval_code_ranker({'dev':splits.get('dev',[])}, lambda e: [r['code'] for r in rows_for(e)])['splits'].get('dev',{})
        dev_rerank=_eval_code_ranker({'dev':splits.get('dev',[])}, rerank_codes)['splits'].get('dev',{})
        fallback=dev_rerank.get('recall@1',0)<dev_base.get('recall@1',0) and dev_rerank.get('mrr',0)<dev_base.get('mrr',0); fallback_used=fallback; dev_before=dev_base; dev_after=dev_rerank; selected_blend_on_dev=tune_blend(splits.get('dev',[]), rows_for, lambda e: rerank_candidates(e.get('context') or e.get('mention'), rows_for(e), reranker, batch_size=int(cfg.get('reranker_batch_size',16))))
        ranker=(lambda e: [r['code'] for r in rows_for(e)]) if fallback else rerank_codes
    if requested != executed: raise ValueError('requested_mode/executed_backend mismatch')
    out={'requested_mode':requested,'executed_backend':executed,'mock_encoder':bool(ns.mock_dense),'mock_reranker':bool(ns.mock_reranker),'model_name':model_name,'model_revision':model_revision,'device':device,'cache_validation':cache_validation,'candidate_universe':candidate_universe,'retrieval_depth':retrieval_depth,'actual_artifact_paths':{'lexical_index_dir':cfg.get('index_dir'),'dense_index_dir':cfg.get('dense_index_dir')},'loaded_lexical_from_cache':getattr(lexical,'loaded_from_cache',False),'lexical_index_load_seconds':getattr(lexical,'index_load_seconds',None),'official_evaluation':False,'synthetic_pilot':True,'readiness':'INVALID_MOCK_RUN' if (ns.mock_dense or ns.mock_reranker) else 'EXPERIMENTAL_SYNTHETIC_ROBUSTNESS_ONLY','max_eval_queries_semantics':'per_split', **_canonical_overlap(records, raw_splits)}
    if selected_on_dev is not None: out['selected_on_dev']=selected_on_dev
    if selected_blend_on_dev is not None: out['selected_blend_on_dev']=selected_blend_on_dev; out['fallback_used']=fallback_used; out['dev_before']=dev_before; out['dev_after']=dev_after
    out.update(_eval_code_ranker(splits, ranker))
    out['dense_backend_called']=dense_called; out['reranker_backend_called']=rerank_called; out['bm25_backend_called']=bm25_called; out.update(counters); out['unique_query_count']=len({_key(e) for rows in splits.values() for e in rows})
    print(json.dumps(out, ensure_ascii=False, indent=2)); return out
if __name__=='__main__': main()
