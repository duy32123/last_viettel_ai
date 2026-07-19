from __future__ import annotations
import argparse, json, math, sys, time
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.kb_schema import KBRecord
from src.linking.retrieval import load_lexical_index, kb_paths_checksum
from src.linking.dense import (
    BGEM3Backend,
    BGERerankerBackend,
    MockDenseEncoder,
    MockReranker,
    dense_expected_manifest,
    load_dense_index,
    metrics_from_ranks,
    rrf_fuse,
    tune_rrf,
    tune_blend,
    tune_candidate_strategy,
    candidate_passage,
)
from scripts.evaluate_hybrid_linking import _encode_query_vectors


def _paths(kb_dir):
    return sorted(Path(kb_dir).glob('*.jsonl')) if Path(kb_dir).exists() else []


def _project_record(d: dict, *, include_relationships: bool = False) -> KBRecord:
    meta = d.get('metadata') or {}
    lean = {'TTY': meta.get('TTY'), 'specificity': meta.get('specificity')}
    if include_relationships:
        lean['relationships'] = meta.get('relationships', [])
    return KBRecord(
        d['code'],
        d['canonical_name'],
        d.get('aliases', []),
        d['terminology'],
        d['version'],
        d['source'],
        d.get('verified', False),
        lean,
        d.get('semantic_type', ''),
        d.get('language', 'en'),
    )


def _load_lightweight(kb_dir, *, include_relationships: bool = False):
    t0 = time.time()
    rows = []
    approx = 0
    for p in _paths(kb_dir):
        with Path(p).open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                rec = _project_record(d, include_relationships=include_relationships)
                rows.append(rec)
                approx += sum(len(str(x)) for x in (rec.code, rec.canonical_name, rec.terminology, rec.version, rec.source, rec.semantic_type, rec.language))
                approx += sum(len(a) for a in rec.aliases)
                approx += 128
    return rows, {
        'lightweight_record_count': len(rows),
        'relationships_loaded': bool(include_relationships),
        'kb_load_seconds': time.time() - t0,
        'approximate_registry_bytes': approx,
    }


def _examples(path):
    d=json.loads(Path(path).read_text())
    return d.get('splits', d)


def _main_examples(splits, max_per_split=0):
    out={}
    for sp,examples in splits.items():
        rows=[e for e in examples if e.get('task')!='exact_alias_diagnostic']
        out[sp]=rows[:max_per_split] if max_per_split else rows
    return out


def _gate_split_leakage(splits):
    by_split=defaultdict(set)
    for sp,examples in splits.items():
        for e in examples:
            by_split[sp].add(e['positive_code'])
    leaks=[]
    names=sorted(by_split)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            overlap=by_split[a] & by_split[b]
            if overlap:
                leaks.append((a,b,sorted(overlap)[:5]))
    if leaks:
        raise ValueError(f'RxCUI split leakage detected: {leaks[:3]}')


def _canonical_overlap(records, splits):
    names={r.code:r.canonical_name.casefold() for r in records}
    total=hit=0
    for examples in splits.values():
        for e in examples:
            if e.get('task')=='exact_alias_diagnostic':
                continue
            total += 1
            name=names.get(e['positive_code'],'').casefold()
            query=(e.get('context') or e.get('mention') or '').casefold()
            if name and name in query:
                hit += 1
    return {'query_contains_canonical_name':hit>0,'canonical_name_overlap_rate':hit/(total or 1)}


def _metrics(ranks):
    return metrics_from_ranks(ranks)


def _eval_code_ranker(splits, ranker):
    out={'splits':{},'per_tty':{},'per_task':{},'candidate_count_distribution':{}}
    all_ranks=[]; tty=defaultdict(list); task=defaultdict(list); counts=Counter(); unresolved=0; total=0
    for sp,examples in splits.items():
        ranks=[]; start=time.time()
        for e in examples:
            codes=ranker(e); counts[len(codes)] += 1; total += 1
            if not codes: unresolved += 1
            rank=codes.index(e['positive_code'])+1 if e['positive_code'] in codes else None
            ranks.append(rank); all_ranks.append(rank); tty[e.get('tty','unknown')].append(rank); task[e.get('task','unknown')].append(rank)
        m=_metrics(ranks); lat=time.time()-start
        m.update({'query_count':len(examples),'latency_sec':lat,'throughput_qps':len(examples)/(lat or 1.0)})
        out['splits'][sp]=m
    out['overall_main_excluding_exact_alias_diagnostic']=_metrics(all_ranks)
    out['per_tty']={k:_metrics(v) for k,v in sorted(tty.items())}
    out['per_task']={k:_metrics(v) for k,v in sorted(task.items())}
    out['candidate_count_distribution']={str(k):v for k,v in sorted(counts.items())}
    out['unresolved_rate']=unresolved/(total or 1)
    return out


def _norm01(values):
    vals=[float(v) for v in values]
    if not vals:
        return []
    lo=min(vals); hi=max(vals)
    if hi == lo:
        return [1.0 for _ in vals]
    return [(v-lo)/(hi-lo) for v in vals]


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--config', default='configs/linking.rxnorm_bge_reranker.yaml')
    p.add_argument('--pilot-examples', default='data/processed/linking_kb_rxnorm/rxnorm_pilot_examples.json')
    p.add_argument('--mode', choices=['bm25','bge_dense','hybrid_rrf','reranker'], default='bm25')
    p.add_argument('--max-eval-queries-per-split', type=int, default=0)
    p.add_argument('--max-eval-queries', type=int, default=0, help='Deprecated global cap alias; treated as per-split for backward compatibility.')
    p.add_argument('--mock-dense', action='store_true')
    p.add_argument('--mock-reranker', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    ns=p.parse_args(argv)
    cfg=json.loads(Path(ns.config).read_text())
    kb_dir=Path(cfg['kb_dir']); paths=_paths(kb_dir); requested=ns.mode
    if ns.dry_run:
        print(json.dumps({'requested_mode':requested,'executed_backend':None,'dry_run':True,'backend_executed':False,'expected_artifacts':{'lexical_index_dir':cfg.get('index_dir'),'dense_index_dir':cfg.get('dense_index_dir')},'official_evaluation':False}, indent=2))
        return

    relationship_expansion=bool(cfg.get('relationship_expansion', False))
    records, load_report = _load_lightweight(kb_dir, include_relationships=relationship_expansion)
    verified_codes={r.code for r in records if r.verified}
    if bool(cfg.get('production',False)) and any(not r.verified for r in records):
        raise ValueError('production RxNorm KB contains unverified records')
    raw_splits=_examples(ns.pilot_examples); _gate_split_leakage(raw_splits)
    cap=ns.max_eval_queries_per_split or ns.max_eval_queries
    splits=_main_examples(raw_splits, cap)
    include_unverified=bool(cfg.get('include_unverified',False))
    top_k=int(cfg.get('top_k',20)); retrieval_depth=int(cfg.get('retrieval_depth',100))
    expected_checksum=kb_paths_checksum(paths); candidate_universe=len(verified_codes)
    counters={'dense_encode_calls':0,'dense_search_calls':0,'bm25_search_calls':0,'ranking_cache_hits':0,'reranker_pairs_scored':0,'reranker_unique_pairs':0,'reranker_model_calls':0,'reranker_batches':0,'reranker_cache_hits':0}
    bm25_called=dense_called=rerank_called=False
    model_name=model_revision=device=None; cache_validation=None
    reranker_model_name=reranker_model_revision=reranker_device=reranker_fp16=None
    selected_on_dev=None; selected_blend_on_dev=None; selected_candidate_strategy=None
    fallback_used=False; dev_before=None; dev_after=None; final_ranking_mode=None
    rank_cache={}; qvec_cache={}

    def _key(e):
        return str(e.get('id') or f"{e.get('positive_code')}:{e.get('task')}:{e.get('context') or e.get('mention')}")

    lexical=load_lexical_index(kb_dir, include_unverified=include_unverified, expected_kb_checksum=expected_checksum, index_dir=Path(cfg.get('index_dir',kb_dir)))

    def bm25_rank(e):
        nonlocal bm25_called
        bm25_called=True; key=('bm25',_key(e))
        if key in rank_cache:
            counters['ranking_cache_hits'] += 1; return rank_cache[key]
        counters['bm25_search_calls'] += 1
        rank_cache[key]=[c.code for c in lexical.search(e.get('context') or e.get('mention'), 'THUỐC', top_k=retrieval_depth, use_fuzzy=False)]
        return rank_cache[key]

    dense_index=encoder=None
    def load_dense():
        nonlocal dense_index,encoder,model_name,model_revision,device,cache_validation
        if dense_index is not None:
            return
        if ns.mock_dense:
            encoder=MockDenseEncoder(); dense_index=__import__('src.linking.dense', fromlist=['DenseAliasIndex']).DenseAliasIndex.build(records, encoder, include_unverified=include_unverified)
            cache_validation='mock'; model_name='mock'; model_revision='mock'; device='mock'; return
        model_name=cfg.get('model_name','BAAI/bge-m3'); model_revision=cfg.get('model_revision','main')
        expected=dense_expected_manifest(cfg, paths, candidate_universe=candidate_universe)
        dense_index=load_dense_index(Path(cfg['dense_index_dir']), expected)
        encoder=BGEM3Backend(model_name=model_name,batch_size=int(cfg.get('batch_size',16)),max_length=int(cfg.get('max_length',8192)),use_fp16=cfg.get('use_fp16'),device=cfg.get('device'))
        device=getattr(encoder,'device',None) or 'auto'; cache_validation='valid'

    def ensure_qvecs():
        load_dense()
        missing_by_key={}
        for examples in splits.values():
            for e in examples:
                missing_by_key.setdefault(_key(e), e)
        missing=[e for k,e in missing_by_key.items() if k not in qvec_cache]
        if not missing:
            return
        bs=max(1,int(cfg.get('batch_size',16)))
        counters['dense_encode_calls'] += math.ceil(len(missing)/bs)
        texts=[e.get('context') or e.get('mention') for e in missing]
        vecs,_=_encode_query_vectors(encoder,texts,batch_size=bs,expected_dim=dense_index._dim())
        for e,v in zip(missing,vecs):
            qvec_cache[_key(e)]=v

    def dense_rank(e):
        nonlocal dense_called
        dense_called=True; ensure_qvecs(); key=('dense',_key(e))
        if key in rank_cache:
            counters['ranking_cache_hits'] += 1; return rank_cache[key]
        counters['dense_search_calls'] += 1
        rank_cache[key]=[r['code'] for r in dense_index.search_vector(qvec_cache[_key(e)], top_k=retrieval_depth)]
        return rank_cache[key]

    def hybrid_rank_from(params):
        return lambda e: rrf_fuse([(params['bm25_weight'],bm25_rank(e)),(params['dense_weight'],dense_rank(e))], params['rrf_k'], top_k=retrieval_depth)

    if requested=='bm25':
        executed='bm25'; ranker=bm25_rank; final_ranking_mode='bm25'
    elif requested=='bge_dense':
        executed='bge_dense'; load_dense(); ensure_qvecs(); ranker=dense_rank; final_ranking_mode='bge_dense'
    elif requested=='hybrid_rrf':
        executed='hybrid_rrf'; load_dense(); ensure_qvecs(); selected_on_dev=tune_rrf(splits.get('dev',[]), bm25_rank, dense_rank); ranker=hybrid_rank_from(selected_on_dev); final_ranking_mode='hybrid_rrf'
    else:
        executed='reranker'; load_dense(); ensure_qvecs()
        selected_on_dev=tune_rrf(splits.get('dev',[]), bm25_rank, dense_rank)
        hybrid_rank=hybrid_rank_from(selected_on_dev)
        selected_candidate_strategy=tune_candidate_strategy(splits.get('dev',[]), dense_rank, hybrid_rank)
        by_code={r.code:r for r in records}
        def selected_strategy_codes(e):
            if selected_candidate_strategy['strategy']=='dense_top20':
                return dense_rank(e)[:20]
            if selected_candidate_strategy['strategy']=='hybrid_top20':
                return hybrid_rank(e)[:20]
            return list(dict.fromkeys(dense_rank(e)[:20] + hybrid_rank(e)[:10]))
        def retrieval_rows(e):
            rows=[]
            for i,code in enumerate(selected_strategy_codes(e)[:20],1):
                rec=by_code[code]
                rows.append({'code':code,'canonical_name':rec.canonical_name,'matched_alias':rec.canonical_name,'score':1.0/i,'rank':i,'verified':rec.verified,'source':rec.source,'version':rec.version})
            return rows
        reranker_model_name=cfg.get('reranker_model_name','BAAI/bge-reranker-v2-m3'); reranker_model_revision=cfg.get('reranker_model_revision','main')
        reranker=MockReranker() if ns.mock_reranker else BGERerankerBackend(reranker_model_name, int(cfg.get('reranker_batch_size',16)), int(cfg.get('reranker_max_length',8192)), cfg.get('use_fp16'), cfg.get('device'))
        reranker_device=getattr(reranker,'device',None) or ('mock' if ns.mock_reranker else 'auto'); reranker_fp16=getattr(reranker,'use_fp16',None)
        score_cache={}
        def score_rows(e, rows):
            nonlocal rerank_called
            q=e.get('context') or e.get('mention')
            missing=[]
            for r in rows:
                k=(_key(e), r['code'])
                if k in score_cache:
                    counters['reranker_cache_hits'] += 1
                else:
                    missing.append((k, q, candidate_passage(r)))
            bs=max(1,int(cfg.get('reranker_batch_size',16)))
            for start in range(0, len(missing), bs):
                batch=missing[start:start+bs]
                pairs=[(q,p) for _,q,p in batch]
                scores=reranker.score(pairs)
                if len(scores) != len(batch):
                    raise ValueError('reranker score count mismatch')
                rerank_called=True
                counters['reranker_model_calls'] += 1; counters['reranker_batches'] += 1; counters['reranker_pairs_scored'] += len(batch)
                for (k,_,_),s in zip(batch,scores):
                    score_cache[k]=float(s)
            counters['reranker_unique_pairs']=len(score_cache)
            return [score_cache[(_key(e), r['code'])] for r in rows]
        def reranked_rows(e):
            rows=retrieval_rows(e); scores=score_rows(e, rows); out=[]
            for r,s in zip(rows,scores):
                row=dict(r); row['reranker_score']=float(s); row['pre_rerank_rank']=row.get('rank'); out.append(row)
            out=sorted(out, key=lambda x:(-x['reranker_score'], x['code']))
            for i,row in enumerate(out,1): row['final_rank']=i
            if {r['code'] for r in rows}!={r['code'] for r in out}: raise ValueError('reranker changed candidate membership')
            return out
        def retrieval_rows_norm(e):
            rows=retrieval_rows(e); norms=_norm01([r.get('score',0.0) for r in rows]); out=[]
            for r,n in zip(rows,norms):
                row=dict(r); row['score']=n; out.append(row)
            return out
        def reranked_rows_norm(e):
            rows=reranked_rows(e); norms=_norm01([r.get('reranker_score',0.0) for r in rows]); out=[]
            for r,n in zip(rows,norms):
                row=dict(r); row['reranker_score']=n; out.append(row)
            return out
        def blended_rows(e, weight):
            rows=retrieval_rows_norm(e); raw_rows=retrieval_rows(e); r_scores=score_rows(e, raw_rows); rer_norm=_norm01(r_scores)
            out=[]
            for r,b in zip(rows,rer_norm):
                row=dict(r); row['reranker_score']=b; row['final_score']=(1-weight)*float(r.get('score',0.0)) + weight*b; out.append(row)
            out=sorted(out, key=lambda x:(-x['final_score'], x['code']))
            for i,row in enumerate(out,1): row['final_rank']=i
            return out
        # Pre-score dev candidates once; tune_blend only reads cached scores.
        for e in splits.get('dev',[]):
            score_rows(e, retrieval_rows(e))
        selected_blend_on_dev=tune_blend(splits.get('dev',[]), retrieval_rows_norm, reranked_rows_norm, weights=tuple(cfg.get('blend_weights',[0.0,0.25,0.5,0.75,1.0])))
        w=float(selected_blend_on_dev['blend_weight'])
        dev_before=_eval_code_ranker({'dev':splits.get('dev',[])}, lambda e: [r['code'] for r in retrieval_rows(e)])['splits'].get('dev',{})
        dev_after=_eval_code_ranker({'dev':splits.get('dev',[])}, lambda e: [r['code'] for r in blended_rows(e,w)])['splits'].get('dev',{})
        fallback_used=dev_after.get('recall@1',0)<dev_before.get('recall@1',0) and dev_after.get('mrr',0)<dev_before.get('mrr',0)
        final_ranking_mode='retrieval_fallback' if fallback_used else f'blended_retrieval_reranker_w={w}'
        ranker=(lambda e: [r['code'] for r in retrieval_rows(e)]) if fallback_used else (lambda e: [r['code'] for r in blended_rows(e,w)])

    if requested != executed:
        raise ValueError('requested_mode/executed_backend mismatch')
    out={'requested_mode':requested,'executed_backend':executed,'mock_encoder':bool(ns.mock_dense),'mock_reranker':bool(ns.mock_reranker),'model_name':model_name,'model_revision':model_revision,'device':device,'cache_validation':cache_validation,'candidate_universe':candidate_universe,'retrieval_depth':retrieval_depth,'actual_artifact_paths':{'lexical_index_dir':cfg.get('index_dir'),'dense_index_dir':cfg.get('dense_index_dir')},'loaded_lexical_from_cache':getattr(lexical,'loaded_from_cache',False),'lexical_index_load_seconds':getattr(lexical,'index_load_seconds',None),'lexical_payload_bytes':getattr(lexical,'lexical_payload_bytes',None),'source_kb_bytes':getattr(lexical,'source_kb_bytes',None),'lexical_compression_ratio':getattr(lexical,'lexical_compression_ratio',None),'official_evaluation':False,'synthetic_pilot':True,'readiness':'INVALID_MOCK_RUN' if (ns.mock_dense or ns.mock_reranker) else 'EXPERIMENTAL_SYNTHETIC_ROBUSTNESS_ONLY','max_eval_queries_semantics':'per_split','final_ranking_mode':final_ranking_mode, **load_report, **_canonical_overlap(records, raw_splits)}
    if selected_on_dev is not None: out['selected_on_dev']=selected_on_dev
    if selected_candidate_strategy is not None: out['selected_candidate_strategy']=selected_candidate_strategy
    if selected_blend_on_dev is not None:
        out.update({'selected_blend_on_dev':selected_blend_on_dev,'fallback_used':fallback_used,'dev_before':dev_before,'dev_after':dev_after,'reranker_model_name':reranker_model_name,'reranker_model_revision':reranker_model_revision,'reranker_device':reranker_device,'reranker_fp16':reranker_fp16})
    out.update(_eval_code_ranker(splits, ranker))
    out['dense_backend_called']=dense_called; out['reranker_backend_called']=rerank_called; out['bm25_backend_called']=bm25_called
    out.update(counters); out['unique_query_count']=len({_key(e) for rows in splits.values() for e in rows})
    print(json.dumps(out, ensure_ascii=False, indent=2)); return out

if __name__=='__main__': main()
