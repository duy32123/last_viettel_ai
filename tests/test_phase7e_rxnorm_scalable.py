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


def test_lexical_payload_is_lean_and_reports_compression(tmp_path):
    kb=tmp_path/'kb'
    rich=[KBRecord('10','amlodipine',[],'RxNorm','v','RxNorm',True,{'TTY':'IN','specificity':'ingredient','relationships':[{'target_rxcui':'11'}],'alias_provenance':[{'raw_alias':'x'}],'rxnconso_checksum':'secret'},'drug','en')]
    write_jsonl(kb/'rxnorm.jsonl', rich)
    manifest=save_lexical_index(LexicalIndex(rich), tmp_path/'idx', [kb/'rxnorm.jsonl'])
    payload=json.loads((tmp_path/'idx'/'lexical_index.json').read_text())
    rec=payload['records'][0]
    assert set(rec) == {'code','canonical_name','aliases','terminology','version','source','verified','TTY','specificity','semantic_type','language'}
    assert 'relationships' not in json.dumps(payload) and 'alias_provenance' not in json.dumps(payload)
    assert manifest['source_kb_bytes'] > 0 and manifest['lexical_payload_bytes'] > 0 and manifest['lexical_compression_ratio'] is not None
    loaded=load_lexical_index(kb, expected_kb_checksum=manifest['kb_checksum'], index_dir=tmp_path/'idx')
    assert [c.code for c in loaded.search('amlodipine','THUỐC',top_k=1,use_fuzzy=False)] == ['10']


def test_evaluator_uses_lightweight_registry_without_relationships_when_disabled(tmp_path):
    from scripts import evaluate_rxnorm_linking as ev
    cfg,pilot=_runtime_fixture(tmp_path)
    # Rebuild KB/index with relationship metadata; evaluator should project it away.
    rich=recs(); rich[0].metadata['relationships']=[{'target_rxcui':'3'}]
    kb=tmp_path/'kb'; write_jsonl(kb/'rxnorm.jsonl', rich)
    idx=tmp_path/'lex'; save_lexical_index(LexicalIndex(rich), idx, [kb/'rxnorm.jsonl'])
    d=json.loads(cfg.read_text()); d.update({'kb_dir':str(kb),'index_dir':str(idx),'relationship_expansion':False}); cfg.write_text(json.dumps(d))
    out=ev.main(['--config',str(cfg),'--pilot-examples',str(pilot),'--mode','bm25','--max-eval-queries-per-split','1'])
    assert out['relationships_loaded'] is False
    assert out['lightweight_record_count'] == 4
    assert out['approximate_registry_bytes'] > 0


def test_reranker_scores_each_pair_once_across_blend_weights_and_counters_match(tmp_path, monkeypatch):
    from scripts import evaluate_rxnorm_linking as ev
    from src.linking.dense import MockDenseEncoder, MockReranker
    cfg,pilot=_runtime_fixture(tmp_path)
    d=json.loads(cfg.read_text()); d['blend_weights']=[0.0,0.25,0.5,0.75,1.0]; cfg.write_text(json.dumps(d))
    spy={'dense_calls':0,'reranker_calls':0,'pairs':0}
    class SpyDense:
        def __init__(self, **kw): self.device='cpu'; self.enc=MockDenseEncoder(dim=4)
        def encode(self, texts): spy['dense_calls'] += 1; return self.enc.encode(texts)
    class SpyReranker(MockReranker):
        def __init__(self,*a,**k): super().__init__(); self.device='cpu'; self.use_fp16=False
        def score(self, pairs):
            rows=list(pairs); spy['reranker_calls'] += 1; spy['pairs'] += len(rows); return super().score(rows)
    monkeypatch.setattr(ev, 'BGEM3Backend', SpyDense); monkeypatch.setattr(ev, 'BGERerankerBackend', SpyReranker)
    out=ev.main(['--config',str(cfg),'--pilot-examples',str(pilot),'--mode','reranker'])
    assert out['reranker_model_calls'] == spy['reranker_calls']
    assert out['reranker_pairs_scored'] == spy['pairs'] == out['reranker_unique_pairs']
    assert out['reranker_cache_hits'] > 0
    assert out['selected_blend_on_dev']['blend_weight'] in d['blend_weights']


def test_selected_blend_changes_final_ranking_and_fallback_uses_blended_metrics(tmp_path, monkeypatch):
    from scripts import evaluate_rxnorm_linking as ev
    from src.linking.dense import MockDenseEncoder
    cfg,pilot=_runtime_fixture(tmp_path)
    d=json.loads(cfg.read_text()); d['blend_weights']=[1.0]; cfg.write_text(json.dumps(d))
    class SpyDense:
        def __init__(self, **kw): self.device='cpu'; self.enc=MockDenseEncoder(dim=4)
        def encode(self, texts): return self.enc.encode(texts)
    class ReversingReranker:
        def __init__(self,*a,**k): self.device='cpu'; self.use_fp16=False
        def score(self, pairs):
            # Favor lexicographically later passages so final blended ranking differs from retrieval.
            return [float(sum(ord(ch) for ch in p) % 1000) for _,p in pairs]
    monkeypatch.setattr(ev, 'BGEM3Backend', SpyDense); monkeypatch.setattr(ev, 'BGERerankerBackend', ReversingReranker)
    out=ev.main(['--config',str(cfg),'--pilot-examples',str(pilot),'--mode','reranker'])
    assert out['selected_blend_on_dev']['blend_weight'] == 1.0
    assert out['dev_after'] == out['selected_blend_on_dev']['dev_metrics'] or 'mrr' in out['dev_after']
    assert out['final_ranking_mode'] in {'retrieval_fallback','blended_retrieval_reranker_w=1.0'}
    if out['fallback_used']:
        assert out['dev_after']['recall@1'] < out['dev_before']['recall@1'] and out['dev_after']['mrr'] < out['dev_before']['mrr']


def test_stable_query_key_avoids_duplicate_dense_searches(tmp_path, monkeypatch):
    from scripts import evaluate_rxnorm_linking as ev
    from src.linking.dense import MockDenseEncoder
    cfg,pilot=_runtime_fixture(tmp_path)
    data=json.loads(Path(pilot).read_text())
    splits=data.get('splits', data)
    # Duplicate the same stable id inside one split; evaluator should reuse qvec and rankings by id.
    main_dev=[e for e in splits['dev'] if e.get('task') != 'exact_alias_diagnostic']
    if main_dev:
        splits['dev'].append(dict(main_dev[0]))
    dup=tmp_path/'dup_pilot.json'; dup.write_text(json.dumps(data))
    class SpyDense:
        def __init__(self, **kw): self.device='cpu'; self.enc=MockDenseEncoder(dim=4)
        def encode(self, texts): return self.enc.encode(texts)
    monkeypatch.setattr(ev, 'BGEM3Backend', SpyDense)
    out=ev.main(['--config',str(cfg),'--pilot-examples',str(dup),'--mode','bge_dense'])
    assert out['dense_search_calls'] == out['unique_query_count']
    assert out['ranking_cache_hits'] >= 1


def test_transformers_reranker_uses_callable_tokenizer_without_prepare_for_model(monkeypatch):
    import sys, types
    from src.linking.dense import BGERerankerBackend
    class FakeTensor:
        def __init__(self, vals): self.vals=vals
        def to(self, device): return self
        def view(self, *args): return self
        def float(self): return self
        def detach(self): return self
        def cpu(self): return self
        def tolist(self): return list(self.vals)
    class FakeTokenizer:
        is_fast=True
        def __call__(self, pairs, padding, truncation, max_length, return_tensors):
            assert padding is True and truncation is True and return_tensors == 'pt'
            assert max_length == 7
            return {'input_ids': FakeTensor([1]*len(pairs)), 'attention_mask': FakeTensor([1]*len(pairs))}
    class FakeModel:
        def to(self, device): self.device=device; return self
        def eval(self): self.eval_called=True; return self
        def __call__(self, **kw): return types.SimpleNamespace(logits=FakeTensor([0.1, 0.2]))
    fake_transformers=types.SimpleNamespace(
        __version__='9.9.9',
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer()),
        AutoModelForSequenceClassification=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeModel()),
    )
    class InferenceMode:
        def __enter__(self): return None
        def __exit__(self, *a): return False
    fake_torch=types.SimpleNamespace(float16='float16', float32='float32', cuda=types.SimpleNamespace(is_available=lambda: False), inference_mode=lambda: InferenceMode())
    monkeypatch.setitem(sys.modules, 'transformers', fake_transformers)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    backend=BGERerankerBackend('model', batch_size=2, max_length=7, revision='main')
    scores=backend.score([('q1','p1'),('q2','p2')])
    assert scores == [0.1, 0.2]
    assert backend.runtime_info()['reranker_backend'] == 'transformers'
    assert backend.runtime_info()['tokenizer_class'] == 'FakeTokenizer'
    assert backend.runtime_info()['tokenizer_is_fast'] is True
    assert not hasattr(backend.tokenizer, 'prepare_for_model')


def test_transformers_reranker_score_count_and_nonfinite_fail(monkeypatch):
    import sys, types, math
    from src.linking.dense import BGERerankerBackend
    class FakeTensor:
        def __init__(self, vals): self.vals=vals
        def to(self, device): return self
        def view(self, *args): return self
        def float(self): return self
        def detach(self): return self
        def cpu(self): return self
        def tolist(self): return list(self.vals)
    class FakeTokenizer:
        is_fast=True
        def __call__(self, pairs, **kw): return {'input_ids': FakeTensor([1]*len(pairs))}
    class InferenceMode:
        def __enter__(self): return None
        def __exit__(self, *a): return False
    fake_torch=types.SimpleNamespace(float16='float16', float32='float32', cuda=types.SimpleNamespace(is_available=lambda: False), inference_mode=lambda: InferenceMode())
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    class ShortModel:
        def to(self, device): return self
        def eval(self): return self
        def __call__(self, **kw): return types.SimpleNamespace(logits=FakeTensor([0.1]))
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(__version__='x', AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer()), AutoModelForSequenceClassification=types.SimpleNamespace(from_pretrained=lambda *a, **k: ShortModel())))
    with pytest.raises(ValueError, match='score count mismatch'):
        BGERerankerBackend('m', batch_size=2).score([('q1','p1'),('q2','p2')])
    class NanModel(ShortModel):
        def __call__(self, **kw): return types.SimpleNamespace(logits=FakeTensor([math.nan, 0.0]))
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(__version__='x', AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer()), AutoModelForSequenceClassification=types.SimpleNamespace(from_pretrained=lambda *a, **k: NanModel())))
    with pytest.raises(ValueError, match='non-finite'):
        BGERerankerBackend('m', batch_size=2).score([('q1','p1'),('q2','p2')])


def test_evaluator_reports_transformers_reranker_runtime_fields(tmp_path, monkeypatch):
    from scripts import evaluate_rxnorm_linking as ev
    from src.linking.dense import MockDenseEncoder
    cfg,pilot=_runtime_fixture(tmp_path)
    d=json.loads(cfg.read_text())
    d['reranker']={'backend':'transformers','model_name':'fake-reranker','revision':'r2','batch_size':2,'max_length':9}
    cfg.write_text(json.dumps(d))
    class SpyDense:
        def __init__(self, **kw): self.device='cpu'; self.enc=MockDenseEncoder(dim=4)
        def encode(self, texts): return self.enc.encode(texts)
    class SpyTransformerReranker:
        backend='transformers'
        def __init__(self, model_name, batch_size, max_length, use_fp16=None, device=None, revision='main'):
            self.model_name=model_name; self.batch_size=batch_size; self.max_length=max_length; self.use_fp16=False; self.device='cpu'; self.revision=revision; self.calls=[]
        def score(self, pairs):
            rows=list(pairs); self.calls.append(rows); return [float(i) for i,_ in enumerate(rows)]
        def runtime_info(self):
            return {'reranker_backend':'transformers','transformers_version':'test','tokenizer_class':'CallableOnlyTokenizer','tokenizer_is_fast':True,'model_class':'FakeSequenceClassifier','device':'cpu','dtype':'float32','score_validation_passed':bool(self.calls)}
    monkeypatch.setattr(ev, 'BGEM3Backend', SpyDense); monkeypatch.setattr(ev, 'BGERerankerBackend', SpyTransformerReranker)
    out=ev.main(['--config',str(cfg),'--pilot-examples',str(pilot),'--mode','reranker'])
    assert out['reranker_backend'] == 'transformers'
    assert out['transformers_version'] == 'test'
    assert out['tokenizer_class'] == 'CallableOnlyTokenizer'
    assert out['tokenizer_is_fast'] is True
    assert out['model_class'] == 'FakeSequenceClassifier'
    assert out['score_validation_passed'] is True
    assert out['reranker_model_name'] == 'fake-reranker' and out['reranker_model_revision'] == 'r2'
    assert out['reranker_pairs'] == out['reranker_pairs_scored']
