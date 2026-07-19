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
    assert load_lexical_index(kb, expected_kb_checksum=manifest['kb_checksum'], index_dir=tmp_path/'index').records
    with pytest.raises(ValueError, match='stale'):
        load_lexical_index(kb, expected_kb_checksum={'rxnorm.jsonl':'bad'}, index_dir=tmp_path/'index')

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
    idx_dir=tmp_path/'index'; save_lexical_index(LexicalIndex(recs()), idx_dir, [kb/'rxnorm.jsonl'])
    cfg={'kb_dir':str(kb),'index_dir':str(idx_dir),'production':True,'include_unverified':False,'top_k':4}
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


def _runtime_fixture(tmp_path):
    from src.linking.dense import DenseAliasIndex, MockDenseEncoder, save_dense_index, dense_expected_manifest
    kb=tmp_path/'kb'; write_jsonl(kb/'rxnorm.jsonl', recs())
    idx_dir=tmp_path/'lex'; save_lexical_index(LexicalIndex(recs()), idx_dir, [kb/'rxnorm.jsonl'])
    dense_dir=tmp_path/'dense'; paths=[kb/'rxnorm.jsonl']
    cfg_dense={'model_name':'test-dense','model_revision':'r1','include_unverified':False,'max_length':32,'batch_size':2}
    manifest=dense_expected_manifest(cfg_dense, paths, dimension=4, candidate_universe=4)
    save_dense_index(DenseAliasIndex.build(recs(), MockDenseEncoder(dim=4), include_unverified=False, manifest=manifest), dense_dir, manifest)
    pilot=tmp_path/'pilot.json'
    from scripts import build_rxnorm_pilot
    build_rxnorm_pilot.main(['--kb-dir',str(kb),'--output',str(pilot),'--max-codes','4','--examples-per-code','5','--seed','5'])
    cfg={'kb_dir':str(kb),'index_dir':str(idx_dir),'dense_index_dir':str(dense_dir),'production':True,'include_unverified':False,'top_k':4,'model_name':'test-dense','model_revision':'r1','max_length':32,'batch_size':2,'reranker_batch_size':1,'reranker_model_name':'test-reranker'}
    cfg_path=tmp_path/'cfg.json'; cfg_path.write_text(json.dumps(cfg))
    return cfg_path,pilot


def test_bge_dense_calls_dense_backend_and_missing_cache_fails(tmp_path, monkeypatch):
    from scripts import evaluate_rxnorm_linking as ev
    from src.linking.dense import MockDenseEncoder
    cfg,pilot=_runtime_fixture(tmp_path); calls={'encode':0}
    class SpyDense:
        def __init__(self, **kw): self.device='cpu'; self.enc=MockDenseEncoder(dim=4)
        def encode(self, texts): calls['encode'] += 1; return self.enc.encode(texts)
    monkeypatch.setattr(ev, 'BGEM3Backend', SpyDense)
    out=ev.main(['--config',str(cfg),'--pilot-examples',str(pilot),'--mode','bge_dense'])
    assert out['executed_backend']=='bge_dense' and out['dense_backend_called'] is True and calls['encode']>0
    d=json.loads(cfg.read_text()); d['dense_index_dir']=str(tmp_path/'missing'); cfg.write_text(json.dumps(d))
    with pytest.raises(FileNotFoundError): ev.main(['--config',str(cfg),'--pilot-examples',str(pilot),'--mode','bge_dense'])


def test_hybrid_calls_bm25_and_dense_and_mock_invalid(tmp_path, monkeypatch):
    from scripts import evaluate_rxnorm_linking as ev
    from src.linking.dense import MockDenseEncoder
    cfg,pilot=_runtime_fixture(tmp_path)
    class SpyDense:
        def __init__(self, **kw): self.device='cpu'; self.enc=MockDenseEncoder(dim=4)
        def encode(self, texts): return self.enc.encode(texts)
    monkeypatch.setattr(ev, 'BGEM3Backend', SpyDense)
    out=ev.main(['--config',str(cfg),'--pilot-examples',str(pilot),'--mode','hybrid_rrf'])
    assert out['executed_backend']=='hybrid_rrf' and out['bm25_backend_called'] and out['dense_backend_called']
    mock=ev.main(['--config',str(cfg),'--pilot-examples',str(pilot),'--mode','hybrid_rrf','--mock-dense'])
    assert mock['readiness']=='INVALID_MOCK_RUN'


def test_reranker_called_membership_safe_and_missing_backend_fails(tmp_path, monkeypatch):
    from scripts import evaluate_rxnorm_linking as ev
    from src.linking.dense import MockDenseEncoder, MockReranker
    cfg,pilot=_runtime_fixture(tmp_path)
    class SpyDense:
        def __init__(self, **kw): self.device='cpu'; self.enc=MockDenseEncoder(dim=4)
        def encode(self, texts): return self.enc.encode(texts)
    class SpyReranker(MockReranker):
        def __init__(self,*a,**k): super().__init__(); self.device='cpu'
    monkeypatch.setattr(ev, 'BGEM3Backend', SpyDense); monkeypatch.setattr(ev, 'BGERerankerBackend', SpyReranker)
    out=ev.main(['--config',str(cfg),'--pilot-examples',str(pilot),'--mode','reranker'])
    assert out['executed_backend']=='reranker' and out['reranker_backend_called'] is True
    class FailReranker:
        def __init__(self,*a,**k): raise RuntimeError('no model')
    monkeypatch.setattr(ev, 'BGERerankerBackend', FailReranker)
    with pytest.raises(RuntimeError, match='no model'):
        ev.main(['--config',str(cfg),'--pilot-examples',str(pilot),'--mode','reranker'])


def test_split_leakage_gate_and_production_excludes_unverified(tmp_path):
    from scripts import evaluate_rxnorm_linking as ev
    cfg,pilot=_runtime_fixture(tmp_path)
    leaked={'train':[{'id':'1:a','positive_code':'1','task':'normalized_medication','context':'x','tty':'IN'}],'dev':[{'id':'1:b','positive_code':'1','task':'normalized_medication','context':'x','tty':'IN'}],'test':[]}
    leak_path=tmp_path/'leak.json'; leak_path.write_text(json.dumps(leaked))
    with pytest.raises(ValueError, match='split leakage'):
        ev.main(['--config',str(cfg),'--pilot-examples',str(leak_path),'--mode','bm25'])
    kb=tmp_path/'badkb'; write_jsonl(kb/'rxnorm.jsonl',[KBRecord('9','fake',[],'RxNorm','v','seed',False,{},'drug','en')])
    idx=tmp_path/'badidx'; save_lexical_index(LexicalIndex([KBRecord('9','fake',[],'RxNorm','v','seed',False,{},'drug','en')], include_unverified=True), idx, [kb/'rxnorm.jsonl'])
    badcfg={'kb_dir':str(kb),'index_dir':str(idx),'production':True,'include_unverified':False}
    bad=tmp_path/'badcfg.json'; bad.write_text(json.dumps(badcfg))
    with pytest.raises(ValueError, match='unverified'):
        ev.main(['--config',str(bad),'--pilot-examples',str(pilot),'--mode','bm25'])


def test_dense_builder_manifest_loads_in_evaluator_and_revision_stale_fails(tmp_path, monkeypatch):
    from scripts import build_dense_linking_index as bd, evaluate_rxnorm_linking as ev
    from src.linking.dense import MockDenseEncoder
    kb=tmp_path/'kb'; write_jsonl(kb/'rxnorm.jsonl', recs())
    idx=tmp_path/'lex'; save_lexical_index(LexicalIndex(recs()), idx, [kb/'rxnorm.jsonl'])
    pilot=tmp_path/'pilot.json'
    from scripts import build_rxnorm_pilot
    build_rxnorm_pilot.main(['--kb-dir',str(kb),'--output',str(pilot),'--max-codes','4','--examples-per-code','2'])
    dense_dir=tmp_path/'dense'
    cfg={'kb_dir':str(kb),'index_dir':str(idx),'dense_index_dir':str(dense_dir),'production':True,'include_unverified':False,'top_k':4,'retrieval_depth':3,'model_name':'test-dense','model_revision':'r1','max_length':32,'batch_size':2}
    cfg_path=tmp_path/'cfg.json'; cfg_path.write_text(json.dumps(cfg))
    class SpyDense:
        def __init__(self, *a, **kw): self.enc=MockDenseEncoder(dim=4); self.device='cpu'
        def encode(self, texts): return self.enc.encode(texts)
    monkeypatch.setattr(bd, 'BGEM3Backend', SpyDense); monkeypatch.setattr(ev, 'BGEM3Backend', SpyDense)
    bd.main.__globals__['sys'].argv=['build_dense_linking_index.py','--config',str(cfg_path)]
    bd.main()
    out=ev.main(['--config',str(cfg_path),'--pilot-examples',str(pilot),'--mode','bge_dense'])
    assert out['cache_validation']=='valid' and out['retrieval_depth']==3
    cfg['model_revision']='stale'; cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='stale dense cache'):
        ev.main(['--config',str(cfg_path),'--pilot-examples',str(pilot),'--mode','bge_dense'])


def test_dense_mmap_and_argpartition_topk_path(tmp_path, monkeypatch):
    import src.linking.dense as dense_mod
    if dense_mod.np is None: pytest.skip('numpy unavailable')
    from src.linking.dense import DenseAliasIndex, MockDenseEncoder, save_dense_index, load_dense_index, dense_expected_manifest
    kb=tmp_path/'kb'; write_jsonl(kb/'rxnorm.jsonl', recs()); paths=[kb/'rxnorm.jsonl']
    manifest=dense_expected_manifest({'model_name':'m','model_revision':'r','include_unverified':False,'max_length':32,'batch_size':2}, paths, dimension=4, candidate_universe=4)
    dense_dir=tmp_path/'dense'; save_dense_index(DenseAliasIndex.build(recs(), MockDenseEncoder(dim=4), False, manifest), dense_dir, manifest)
    calls={'load':0,'argpartition':0}
    real_load=dense_mod.np.load; real_arg=dense_mod.np.argpartition
    def spy_load(*args, **kwargs):
        assert kwargs.get('mmap_mode')=='r'; calls['load']+=1; return real_load(*args, **kwargs)
    def spy_arg(*args, **kwargs): calls['argpartition']+=1; return real_arg(*args, **kwargs)
    monkeypatch.setattr(dense_mod.np, 'load', spy_load); monkeypatch.setattr(dense_mod.np, 'argpartition', spy_arg)
    idx=load_dense_index(dense_dir, manifest)
    idx.search_vector(MockDenseEncoder(dim=4).encode(['metformin'])[0], top_k=2)
    assert calls['load']==1 and calls['argpartition']>=1
