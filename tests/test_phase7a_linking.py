import json, pytest
from pathlib import Path
from src.data.import_icd import import_icd_official
from src.data.import_rxnorm import import_rxnorm_rrf
from src.data.kb_schema import KBRecord, alias_collisions
from src.linking.normalization import normalize_mention
from src.linking.retrieval import LexicalIndex
from src.linking.pipeline import link_rows, validate_kb
from src.linking.metrics import evaluate

def ent(text, mention, typ):
    s=text.index(mention); return {'text':mention,'type':typ,'position':[s,s+len(mention)],'assertions':[]}

def test_normalization_unicode_no_diacritic_strength_and_offsets():
    n=normalize_mention('Metformin 500 mg','THUỐC')
    assert n['base']=='metformin' and n['features']['strength']=='500 mg'
    assert normalize_mention('ĐTĐ')['no_diacritic']=='đái tháo đường' or normalize_mention('ĐTĐ')['normalized']=='đái tháo đường'

def test_rxnorm_rrf_parsing_suppress_and_routing():
    recs=import_rxnorm_rrf(Path('tests/fixtures/linking/rxnorm_rrf'),'TEST','fixture')
    codes={r.code for r in recs}; assert '6809' in codes and '607999' in codes and '1111' not in codes
    idx=LexicalIndex(recs)
    assert idx.search('Metformin 500 mg','THUỐC')[0].code=='6809'
    assert idx.search('Glucophage','THUỐC')[0].code=='607999'
    assert idx.search('HbA1c','TÊN_XÉT_NGHIỆM') == []

def test_icd10_vs_icd10cm_gate_and_alias_collision(tmp_path):
    p=tmp_path/'icd.csv'; p.write_text('code,title,aliases,version,language,terminology\nE11,Diabetes,ĐTĐ,v,vi,ICD-10-CM\n', encoding='utf-8')
    with pytest.raises(ValueError): import_icd_official(p,'ICD-10','v','fixture')
    a=KBRecord('E11','A',['same'],'ICD-10','v','fixture',True); b=KBRecord('I10','B',['same'],'ICD-10','v','fixture',True)
    assert alias_collisions([a,b])

def test_linking_contract_candidate_missing_metrics_and_gate():
    icd=import_icd_official(Path('tests/fixtures/linking/icd10_fixture.csv'),'ICD-10','TEST-2026','fixture')
    rx=import_rxnorm_rrf(Path('tests/fixtures/linking/rxnorm_rrf'),'TEST-2026','fixture')
    idx=LexicalIndex(icd+rx)
    text='Bệnh nhân ĐTĐ dùng Metformin 500 mg. HbA1c 8.1%.'
    rows=[{'id':'d1','text':text,'entities':[ent(text,'ĐTĐ','CHẨN_ĐOÁN'), ent(text,'Metformin 500 mg','THUỐC'), ent(text,'HbA1c','TÊN_XÉT_NGHIỆM')]}]
    out=link_rows(rows, idx, top_k=5); ents=out[0]['entities']
    assert ents[0]['candidates'][0]['code']=='E11' and ents[0]['candidates'][0]['verified'] is True
    assert ents[1]['candidates'][0]['code']=='6809'
    assert ents[2]['candidates']==[] and ents[2]['candidate_missing'] is True
    gold=[{'id':'d1','entities':[{**ents[0],'gold_code':'E11'},{**ents[1],'gold_code':'6809'}]}]
    m=evaluate(gold,out); assert m['recall@1']==1.0 and m['mrr']==1.0
    with pytest.raises(ValueError): validate_kb([], production=True)
    with pytest.raises(ValueError): validate_kb([KBRecord('E11','Diabetes',[],'ICD-10','v','fixture',False)], production=True)

def test_mocked_embedding_reranker_interfaces_not_required():
    class MockEmbedder:
        def score(self, q, docs): return [1.0 for _ in docs]
    class MockReranker:
        def rerank(self, q, cands): return cands
    assert MockReranker().rerank('q',[1]) == [1] and MockEmbedder().score('q',['d']) == [1.0]
