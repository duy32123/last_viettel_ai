from pathlib import Path
import argparse, json, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.kb_schema import read_jsonl
from src.data.import_hf_icd_aux import make_pilot_examples, assert_no_query_kb_leakage, evaluate_pilot_bm25
from src.linking.retrieval import LexicalIndex
from src.linking.dense import DenseAliasIndex, MockDenseEncoder, tune_rrf, rrf_fuse, metrics_from_ranks

def _rows_from_pilot(path):
 d=json.loads(Path(path).read_text()); return d

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/linking.bge_m3_pilot.yaml'); p.add_argument('--pilot-examples', default='data/processed/linking_kb_auxiliary/auxiliary_pilot_examples.json'); p.add_argument('--mode', choices=['bm25','bge_dense','hybrid_rrf'], default=None); ns=p.parse_args(); cfg=json.loads(Path(ns.config).read_text())
 records=[]
 for pth in Path(cfg['kb_dir']).glob('*.jsonl'): records.extend(read_jsonl(pth))
 splits=_rows_from_pilot(ns.pilot_examples); assert_no_query_kb_leakage(records, splits)
 bm25=LexicalIndex(records, include_unverified=bool(cfg.get('include_unverified',False))); encoder=MockDenseEncoder(); dense=DenseAliasIndex.build(records, encoder, bool(cfg.get('include_unverified',False)))
 def bm_rank(ex): return [c.code for c in bm25.search(ex['journal_note'], 'CHẨN_ĐOÁN', top_k=len(records), use_fuzzy=False)]
 def de_rank(ex): return [c['code'] for c in dense.search_with_encoder(ex['journal_note'], encoder, top_k=len(records))]
 params=tune_rrf(splits.get('dev',[]), bm_rank, de_rank) if splits.get('dev') else {'bm25_weight':1.0,'dense_weight':1.0,'rrf_k':60}
 mode=ns.mode or cfg.get('mode','hybrid_rrf'); out={'official_evaluation':False,'task':'note_to_code_auxiliary','retrieval_mode':mode,'candidate_count':len(records),'selected_on_dev':params,'splits':{}}
 for split,examples in splits.items():
  t=time.time(); ranks=[]
  for ex in examples:
   if mode=='bm25': codes=bm_rank(ex)
   elif mode=='bge_dense': codes=de_rank(ex)
   else: codes=rrf_fuse([(params['bm25_weight'],bm_rank(ex)),(params['dense_weight'],de_rank(ex))], params['rrf_k'], top_k=len(records))
   ranks.append(codes.index(ex['positive_code'])+1 if ex['positive_code'] in codes else None)
  m=metrics_from_ranks(ranks); m['latency_sec']=time.time()-t; m['throughput_qps']=(len(examples)/(m['latency_sec'] or 1)); out['splits'][split]=m
 allr=[]
 for split,examples in splits.items():
  for ex in examples:
   codes=bm_rank(ex) if mode=='bm25' else (de_rank(ex) if mode=='bge_dense' else rrf_fuse([(params['bm25_weight'],bm_rank(ex)),(params['dense_weight'],de_rank(ex))], params['rrf_k'], top_k=len(records)))
   allr.append(codes.index(ex['positive_code'])+1 if ex['positive_code'] in codes else None)
 out['overall']=metrics_from_ranks(allr); out['readiness']='READY_FOR_RERANKER' if out['overall']['recall@10']>=0.80 and out['overall']['recall@20']>=0.90 else 'NOT_READY_FOR_RERANKER'
 print(json.dumps(out, ensure_ascii=False, indent=2))
if __name__=='__main__': main()
