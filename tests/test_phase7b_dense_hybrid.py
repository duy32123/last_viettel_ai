import json, pytest
from pathlib import Path
from src.data.kb_schema import KBRecord
from src.linking.dense import DenseAliasIndex, MockDenseEncoder, l2_normalize, dot, save_dense_index, load_dense_index, rrf_fuse, tune_rrf
from src.linking.retrieval import LexicalIndex

def records():
    return [KBRecord('E11','Diabetes',['Đái tháo đường'],'ICD-10','v','aux',False), KBRecord('I10','Hypertension',['Tăng huyết áp'],'ICD-10','v','aux',False), KBRecord('J45','Asthma',['Hen phế quản'],'ICD-10','v','aux',False)]

def test_vector_normalization_alias_collapse_and_tie_order():
    recs=records(); enc=MockDenseEncoder(dim=4); idx=DenseAliasIndex.build(recs, enc, include_unverified=True)
    assert all(abs(sum(x*x for x in v)-1.0)<1e-6 for v in idx.vectors)
    res=idx.search_with_encoder('Đái tháo đường', enc, top_k=3)
    assert res[0]['code']=='E11' and res[0]['rank']==1 and res[0]['matched_alias'] in {'Diabetes','Đái tháo đường'}
    zero=[1.0,0.0,0.0,0.0]; tied=DenseAliasIndex.build(recs, enc, include_unverified=True); tied.vectors=[zero for _ in tied.vectors]
    assert [r['code'] for r in tied.search_vector(zero, top_k=3)] == ['E11','I10','J45']

def test_cache_invalidation(tmp_path):
    recs=records(); enc=MockDenseEncoder(); manifest={'model_name':'BAAI/bge-m3','kb_checksum':'abc','dimension':4,'normalization':'l2'}
    idx=DenseAliasIndex.build(recs, enc, include_unverified=True, manifest=manifest); save_dense_index(idx,tmp_path,manifest)
    assert load_dense_index(tmp_path, {'model_name':'BAAI/bge-m3','kb_checksum':'abc'}).manifest['kb_checksum']=='abc'
    with pytest.raises(ValueError): load_dense_index(tmp_path, {'model_name':'BAAI/bge-m3','kb_checksum':'stale'})

def test_rrf_and_dev_only_tuning_not_test():
    dev=[{'journal_note':'a','positive_code':'E11'},{'journal_note':'b','positive_code':'I10'}]
    test=[{'journal_note':'c','positive_code':'J45'}]
    def bm(ex): return ['E11','I10','J45']
    def de(ex): return ['I10','E11','J45']
    assert rrf_fuse([(1.0,bm(dev[0])),(1.0,de(dev[0]))], k=60, top_k=2) == ['E11','I10']
    params=tune_rrf(dev,bm,de); params2=tune_rrf(dev,bm,lambda ex:['J45','I10','E11'])
    assert 'dev_metrics' in params and params == tune_rrf(dev,bm,de)
    assert params != tune_rrf(test,bm,de)

def test_alias_holdout_and_runtime_schema_verified_default():
    recs=records(); prod=LexicalIndex(recs); aux=LexicalIndex(recs, include_unverified=True)
    assert prod.search('Diabetes','CHẨN_ĐOÁN') == []
    assert aux.search('Diabetes','CHẨN_ĐOÁN')[0].code == 'E11'
    idx=DenseAliasIndex.build(recs, MockDenseEncoder(), include_unverified=True)
    c=idx.search_with_encoder('Diabetes', MockDenseEncoder(), top_k=1)[0]
    assert {'code','canonical_name','terminology','score','rank','retrieval_method','matched_alias','verified','source'} <= set(c)


def _write_kb_and_pilot(tmp_path, recs=None):
    from src.data.kb_schema import write_jsonl
    recs = recs or records()
    kb_dir = tmp_path / 'kb'; write_jsonl(kb_dir / 'icd10.jsonl', recs)
    pilot = {
        'train': [{'id':'tr1','journal_note':'diabetes note','positive_code':'E11','language':'en'}],
        'dev': [{'id':'dv1','journal_note':'hypertension note','positive_code':'I10','language':'en'}],
        'test': [{'id':'ts1','journal_note':'asthma note','positive_code':'J45','language':'vi'}],
    }
    pilot_path = tmp_path / 'pilot.json'; pilot_path.write_text(json.dumps(pilot), encoding='utf-8')
    return kb_dir, pilot_path


def test_dense_search_requires_explicit_encoder_or_query_vector():
    idx = DenseAliasIndex.build(records(), MockDenseEncoder(), include_unverified=True)
    with pytest.raises(RuntimeError):
        idx.search('Diabetes')
    with pytest.raises(ValueError, match='dimension mismatch'):
        idx.search_vector(l2_normalize([1.0, 0.0]), top_k=1)


def test_bm25_full_ranking_returns_full_candidate_universe():
    from scripts.evaluate_hybrid_linking import full_bm25_rank
    idx = LexicalIndex(records(), include_unverified=True)
    assert full_bm25_rank(idx, 'zzzz unmatched query', ['E11', 'I10', 'J45']) == ['E11', 'I10', 'J45']
    assert len(full_bm25_rank(idx, 'Diabetes', ['E11', 'I10', 'J45'])) == 3


def test_evaluator_explicit_mock_mode_is_invalid_and_non_hybrid_has_no_selected_on_dev(tmp_path, capsys):
    from scripts.evaluate_hybrid_linking import main
    kb_dir, pilot_path = _write_kb_and_pilot(tmp_path)
    cfg = {'kb_dir': str(kb_dir), 'dense_index_dir': str(tmp_path/'dense'), 'include_unverified': True, 'mode': 'bge_dense', 'model_name': 'test-model', 'model_revision': 'r1', 'max_length': 32, 'batch_size': 2}
    cfg_path = tmp_path / 'cfg.json'; cfg_path.write_text(json.dumps(cfg), encoding='utf-8')
    out = main(['--config', str(cfg_path), '--pilot-examples', str(pilot_path), '--mode', 'bge_dense', '--mock-dense'])
    assert out['mock_encoder'] is True
    assert out['readiness'] == 'INVALID_MOCK_RUN'
    assert 'selected_on_dev' not in out
    assert out['dense_preflight']['used_mock_encoder'] is True


def test_normal_evaluator_loads_persisted_vectors_and_fails_when_stale(tmp_path, monkeypatch):
    from src.data.kb_schema import write_jsonl
    from src.linking.dense import kb_checksum
    from scripts import evaluate_hybrid_linking as ev
    recs = records(); kb_dir, pilot_path = _write_kb_and_pilot(tmp_path, recs)
    checksum = kb_checksum([kb_dir / 'icd10.jsonl'])
    dense_dir = tmp_path / 'dense'
    manifest = {'model_name':'test-model','model_revision':'r1','kb_checksum':checksum,'normalization':'l2','include_unverified':True,'max_length':32,'dimension':4}
    idx = DenseAliasIndex.build(recs, MockDenseEncoder(), include_unverified=True, manifest=manifest)
    save_dense_index(idx, dense_dir, manifest)
    cfg = {'kb_dir': str(kb_dir), 'dense_index_dir': str(dense_dir), 'include_unverified': True, 'mode': 'bge_dense', 'model_name': 'test-model', 'model_revision': 'r1', 'max_length': 32, 'batch_size': 2}
    cfg_path = tmp_path / 'cfg.json'; cfg_path.write_text(json.dumps(cfg), encoding='utf-8')
    class FakeBackend:
        def __init__(self, **kwargs): self.device='cpu'; self.enc=MockDenseEncoder(dim=4)
        def encode(self, texts): return self.enc.encode(texts)
    monkeypatch.setattr(ev, 'BGEM3Backend', FakeBackend)
    out = ev.main(['--config', str(cfg_path), '--pilot-examples', str(pilot_path), '--mode', 'bge_dense'])
    assert out['mock_encoder'] is False
    assert out['dense_preflight']['cache_validation_result'] == 'valid'
    assert out['dense_preflight']['query_vector_count'] == 3
    assert out['diagnostic_exact_alias']['query_vector_count'] == 6
    cfg['model_revision'] = 'stale'; cfg_path.write_text(json.dumps(cfg), encoding='utf-8')
    with pytest.raises(ValueError, match='stale dense cache'):
        ev.main(['--config', str(cfg_path), '--pilot-examples', str(pilot_path), '--mode', 'bge_dense'])


def test_batch_encoding_dimension_mismatch_fails(tmp_path):
    from scripts.evaluate_hybrid_linking import _encode_query_vectors
    class BadEncoder:
        def encode(self, texts): return [l2_normalize([1.0, 0.0]) for _ in texts]
    with pytest.raises(ValueError, match='dimension mismatch'):
        _encode_query_vectors(BadEncoder(), ['a','b','c'], batch_size=2, expected_dim=4)


def test_rrf_weight_zero_supported_but_both_zero_rejected():
    dev=[{'journal_note':'a','positive_code':'E11'}]
    def bm(ex): return ['E11','I10','J45']
    def de(ex): return ['J45','I10','E11']
    params = tune_rrf(dev, bm, de, weights=(0.0, 1.0), ks=(10,))
    assert (params['bm25_weight'], params['dense_weight']) != (0.0, 0.0)
    assert params['bm25_weight'] in {0.0, 1.0}


def test_hybrid_selected_on_dev_and_readiness_uses_test_only(tmp_path, monkeypatch):
    from src.linking.dense import kb_checksum
    from scripts import evaluate_hybrid_linking as ev
    recs=records(); kb_dir, pilot_path = _write_kb_and_pilot(tmp_path, recs)
    checksum = kb_checksum([kb_dir / 'icd10.jsonl'])
    dense_dir = tmp_path/'dense'
    manifest={'model_name':'test-model','model_revision':'r1','kb_checksum':checksum,'normalization':'l2','include_unverified':True,'max_length':32,'dimension':4}
    save_dense_index(DenseAliasIndex.build(recs, MockDenseEncoder(), include_unverified=True, manifest=manifest), dense_dir, manifest)
    cfg={'kb_dir':str(kb_dir),'dense_index_dir':str(dense_dir),'include_unverified':True,'mode':'hybrid_rrf','model_name':'test-model','model_revision':'r1','max_length':32,'batch_size':2}
    cfg_path=tmp_path/'cfg.json'; cfg_path.write_text(json.dumps(cfg), encoding='utf-8')
    class FakeBackend:
        def __init__(self, **kwargs): self.device='cpu'; self.enc=MockDenseEncoder(dim=4)
        def encode(self, texts): return self.enc.encode(texts)
    monkeypatch.setattr(ev, 'BGEM3Backend', FakeBackend)
    out=ev.main(['--config',str(cfg_path),'--pilot-examples',str(pilot_path),'--mode','hybrid_rrf'])
    assert 'selected_on_dev' in out
    assert out['readiness'] in {'NOT_READY_FOR_RERANKER','READY_FOR_RERANKER'}
    if out['splits']['test']['recall@10'] < 0.80:
        assert out['readiness'] == 'NOT_READY_FOR_RERANKER'
