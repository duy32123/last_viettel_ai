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
