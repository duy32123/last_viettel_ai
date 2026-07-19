import json
from pathlib import Path
import pytest
from src.data.kb_schema import KBRecord, write_jsonl
from src.linking.retrieval import LexicalIndex, save_lexical_index, load_lexical_index, expand_relationship_candidates

def recs():
    return [
        KBRecord('1','metformin',[],'RxNorm','v','RxNorm',True,{'TTY':'IN','specificity':'ingredient'},'drug','en'),
        KBRecord('2','Glucophage',[],'RxNorm','v','RxNorm',True,{'TTY':'BN','specificity':'brand'},'drug','en'),
        KBRecord('3','metformin 500 MG Oral Tablet',[],'RxNorm','v','RxNorm',True,{'TTY':'SCD','specificity':'product'},'drug','en'),
        KBRecord('4','insulin',[],'RxNorm','v','RxNorm',True,{'TTY':'PIN','specificity':'ingredient'},'drug','en'),
    ]

def test_inverted_bm25_matches_reference_scan_and_no_query_full_scan():
    idx=LexicalIndex(recs())
    query='metformin 500 mg đường uống'
    fast=[c.code for c in idx.search(query,'THUỐC',top_k=4,use_fuzzy=False)]
    # A query should not need self.names in the BM25 path once postings are built.
    class Bomb:
        def __iter__(self): raise AssertionError('full scan used')
    idx.names=Bomb()
    fast_no_scan=[c.code for c in idx.search(query,'THUỐC',top_k=4,use_fuzzy=False)]
    assert fast_no_scan == fast
    assert '3' in fast_no_scan and len(idx.records)==4

def test_lexical_index_persist_manifest_and_stale_cache(tmp_path):
    kb=tmp_path/'kb'; write_jsonl(kb/'rxnorm.jsonl', recs())
    idx=LexicalIndex(recs()); manifest=save_lexical_index(idx,tmp_path/'index',[kb/'rxnorm.jsonl'],{'production':True})
    assert manifest['candidate_universe']==4 and manifest['index_size']['postings']>0
    assert load_lexical_index(kb, expected_kb_checksum=manifest['kb_checksum']).records
    with pytest.raises(ValueError, match='stale'):
        load_lexical_index(kb, expected_kb_checksum={'rxnorm.jsonl':'bad'})

def test_deterministic_stratified_sampling_and_split_leakage_zero():
    from scripts.build_rxnorm_pilot import deterministic_sample, split_code, stratum, context_for
    sample1=deterministic_sample(recs(), max_codes=3, seed=7)
    sample2=deterministic_sample(list(reversed(recs())), max_codes=3, seed=7)
    assert [r.code for r in sample1] == [r.code for r in sample2]
    assert {'ingredient','brand','product'} & {stratum(r) for r in sample1}
    code_splits={r.code:split_code(r.code,7) for r in sample1}
    assert len(code_splits)==len(sample1)
    assert all(ex['task'] for r in sample1 for ex in context_for(r, examples_per_code=5))

def test_exact_diagnostic_excluded_from_main_metric(tmp_path, capsys):
    from scripts import build_rxnorm_pilot, evaluate_rxnorm_linking
    kb=tmp_path/'kb'; write_jsonl(kb/'rxnorm.jsonl', recs())
    pilot=tmp_path/'pilot.json'
    build_rxnorm_pilot.main(['--kb-dir',str(kb),'--output',str(pilot),'--max-codes','4','--examples-per-code','5','--seed','5'])
    cfg={'kb_dir':str(kb),'production':True,'include_unverified':False,'top_k':4}
    cfg_path=tmp_path/'cfg.json'; cfg_path.write_text(json.dumps(cfg))
    out=evaluate_rxnorm_linking.main(['--config',str(cfg_path),'--pilot-examples',str(pilot),'--mode','bm25','--max-eval-queries','2'])
    assert 'overall_main_excluding_exact_alias_diagnostic' in out
    assert 'exact_alias_diagnostic' not in out.get('per_task',{})
    assert out['candidate_universe']==4

def test_relationship_expansion_provenance_configurable():
    base=recs(); base[0].metadata['relationships']=[{'target_rxcui':'3','rela':'has_tradename','source':'RXNREL.RRF'}]
    cand=[{'code':'1','canonical_name':'metformin','score':1.0,'rank':1,'retrieval_method':'exact','matched_alias':'metformin','verified':True,'source':'RxNorm','version':'v'}]
    assert expand_relationship_candidates(cand, base, enabled=False)==cand
    expanded=expand_relationship_candidates(cand, base, enabled=True)
    assert [c['code'] for c in expanded]==['1','3']
    assert expanded[1]['relationship_provenance']['source']=='RXNREL.RRF'

def test_candidate_membership_after_mock_rerank_unchanged():
    from src.linking.dense import MockReranker, rerank_candidates
    rows=[{'code':'1','canonical_name':'metformin','matched_alias':'metformin','score':1,'rank':1},{'code':'2','canonical_name':'Glucophage','matched_alias':'Glucophage','score':.5,'rank':2}]
    out=rerank_candidates('Glucophage', rows, MockReranker({('Glucophage','Glucophage'):2.0}))
    assert {r['code'] for r in out} == {'1','2'}
