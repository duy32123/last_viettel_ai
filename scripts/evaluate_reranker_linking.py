from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_hybrid_linking import _load_records, _rows_from_pilot, _load_dense_for_eval, _encode_query_vectors, full_bm25_rank, _evaluate_ranker
from src.data.import_hf_icd_aux import assert_no_query_kb_leakage
from src.linking.dense import BGERerankerBackend, MockReranker, metrics_from_ranks, rrf_fuse, rerank_candidates, tune_candidate_strategy, tune_blend
from src.linking.retrieval import LexicalIndex

DEFAULT_RRF={'bm25_weight':0.5,'dense_weight':1.0,'rrf_k':10}

def _rank_metrics_from_rows(splits, row_ranker):
    return _evaluate_ranker(splits, lambda ex:[r['code'] for r in row_ranker(ex)])

def _candidate_recall(examples, candidate_fn):
    vals=[]
    for ex in examples:
        vals.append(ex['positive_code'] in [r['code'] for r in candidate_fn(ex)])
    return sum(vals)/(len(vals) or 1)

def _make_candidate_rows(records, code_order, dense_by_code):
    by_code={r.code:r for r in records}
    rows=[]
    for rank,code in enumerate(code_order,1):
        if code in dense_by_code:
            row=dict(dense_by_code[code]); row.setdefault('retrieval_rank', rank); row['rank']=rank; rows.append(row); continue
        rec=by_code[code]
        rows.append({'code':rec.code,'canonical_name':rec.canonical_name,'terminology':rec.terminology,'score':0.0,'rank':rank,'retrieval_rank':rank,'retrieval_method':'bm25_zero_fill','matched_alias':rec.canonical_name,'verified':rec.verified,'source':rec.source,'version':rec.version})
    return rows

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--config', default='configs/linking.bge_m3_pilot.yaml'); ap.add_argument('--pilot-examples', default='data/processed/linking_kb_auxiliary/auxiliary_pilot_examples.json'); ap.add_argument('--mock-dense', action='store_true'); ap.add_argument('--mock-reranker', action='store_true'); ns=ap.parse_args(argv)
    cfg=json.loads(Path(ns.config).read_text(encoding="utf-8"))
    records,kb_paths=_load_records(cfg['kb_dir']); splits=_rows_from_pilot(ns.pilot_examples); leak=assert_no_query_kb_leakage(records,splits)
    include_unverified=bool(cfg.get('include_unverified',False)); all_codes=sorted({r.code for r in records if r.verified or include_unverified})
    bm25=LexicalIndex(records, include_unverified=include_unverified)
    dense,encoder,preflight=_load_dense_for_eval(cfg, records, kb_paths, mock_dense=ns.mock_dense)
    keyed={sp:[{**row,'_eval_key':f'{sp}:{row["id"]}'} for row in rows] for sp,rows in splits.items()}
    texts=[]; keys=[]
    for sp,rows in splits.items():
        for row in rows: texts.append(row['journal_note']); keys.append(f'{sp}:{row["id"]}')
    qvecs,timing=_encode_query_vectors(encoder,texts,batch_size=int(cfg.get('batch_size',16)),expected_dim=preflight['index_dimension']); vectors=dict(zip(keys,qvecs)); preflight.update(timing)
    def bm_rank(ex): return full_bm25_rank(bm25, ex['journal_note'], all_codes)
    def dense_rows(ex): return dense.search_vector(vectors[ex['_eval_key']], top_k=len(all_codes))
    def dense_rank(ex): return [r['code'] for r in dense_rows(ex)]
    params=cfg.get('rrf_params') or DEFAULT_RRF
    def hybrid_rank(ex): return rrf_fuse([(params['bm25_weight'],bm_rank(ex)),(params['dense_weight'],dense_rank(ex))], params['rrf_k'], top_k=len(all_codes))
    selected=tune_candidate_strategy(keyed.get('dev',[]), dense_rank, hybrid_rank)
    def selected_codes(ex):
        if selected['strategy']=='dense_top20': return dense_rank(ex)[:20]
        if selected['strategy']=='hybrid_top20': return hybrid_rank(ex)[:20]
        return list(dict.fromkeys(dense_rank(ex)[:20]+hybrid_rank(ex)[:10]))
    def original_rows(ex):
        dby={r['code']:r for r in dense_rows(ex)}
        return _make_candidate_rows(records, selected_codes(ex), dby)
    reranker = MockReranker() if ns.mock_reranker else BGERerankerBackend(cfg.get('reranker_model_name','BAAI/bge-reranker-v2-m3'), int(cfg.get('reranker_batch_size',16)), int(cfg.get('reranker_max_length',8192)), cfg.get('use_fp16'), cfg.get('device'))
    pair_count=0; start=time.time()
    def reranked_rows(ex):
        nonlocal pair_count
        rows=original_rows(ex); pair_count += len(rows)
        out=rerank_candidates(ex['journal_note'], rows, reranker, batch_size=int(cfg.get('reranker_batch_size',16)))
        if {r['code'] for r in rows}!={r['code'] for r in out}: raise ValueError('reranker changed candidate membership')
        return out
    original=_rank_metrics_from_rows(keyed, original_rows); reranked=_rank_metrics_from_rows(keyed, reranked_rows)
    blend=tune_blend(keyed.get('dev',[]), original_rows, reranked_rows, tuple(cfg.get('blend_weights',[0.0,0.25,0.5,0.75,1.0])))
    dev_o=original['splits'].get('dev',{}); dev_r=reranked['splits'].get('dev',{})
    fallback = dev_r.get('mrr',0)<dev_o.get('mrr',0) and dev_r.get('recall@1',0)<dev_o.get('recall@1',0)
    out={'official_evaluation':False,'task':'note_to_code_auxiliary_rerank','query_kb_overlap':leak['query_kb_overlap'],'mock_dense':bool(ns.mock_dense),'mock_reranker':bool(ns.mock_reranker),'dense_preflight':preflight,'selected_candidate_strategy':selected,'rrf_params':params,'original_candidate_ranking':original,'reranker_only':reranked,'selected_blend_on_dev':blend,'fallback_to_phase7b':fallback,'candidate_recall':{sp:_candidate_recall(rows, original_rows) for sp,rows in keyed.items()},'candidate_membership_unchanged':True,'latency_sec':time.time()-start,'pair_count':pair_count,'pair_throughput_per_sec':pair_count/((time.time()-start) or 1.0),'peak_vram':None,'readiness':'INVALID_MOCK_RUN' if ns.mock_reranker or ns.mock_dense else ('FALLBACK_TO_PHASE7B' if fallback else 'RERANKER_EVALUATED_NOT_OFFICIAL')}
    print(json.dumps(out, ensure_ascii=False, indent=2)); return out
if __name__=='__main__': main()

