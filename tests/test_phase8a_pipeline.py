import json
from pathlib import Path
import pytest
from src.pipeline.end_to_end import EndToEndPipeline, load_config

FIXTURE_TEXT = "Không ghi nhận đau ngực. Tiền sử gia đình có tăng huyết áp.\nNgười bệnh dùng metformin 500 mg.\nXét nghiệm HbA1c 8.1%.\n"
ALLOWED={"text","type","position","assertions","candidates"}


def cfg():
    return load_config('configs/pipeline.end_to_end.smoke.yaml')


def test_end_to_end_fixture_schema_offsets_and_candidates():
    pipe=EndToEndPipeline(cfg())
    out=pipe.infer_document(FIXTURE_TEXT)
    assert all(set(row) <= ALLOWED for row in out)
    assert all(FIXTURE_TEXT[row['position'][0]:row['position'][1]] == row['text'] for row in out)
    by_text={row['text']:row for row in out}
    assert by_text['đau ngực']['type'] == 'TRIỆU_CHỨNG'
    assert by_text['đau ngực']['assertions'] == ['isNegated']
    assert by_text['tăng huyết áp']['type'] == 'CHẨN_ĐOÁN'
    assert by_text['tăng huyết áp']['assertions'] == ['isFamily','isHistorical']
    assert by_text['tăng huyết áp']['candidates'] == []
    assert by_text['metformin 500 mg']['type'] == 'THUỐC'
    assert by_text['metformin 500 mg']['candidates'] == ['6809']
    assert by_text['HbA1c']['type'] == 'TÊN_XÉT_NGHIỆM'
    assert by_text['8.1%']['type'] == 'KẾT_QUẢ_XÉT_NGHIỆM'
    assert 'relations' not in json.dumps(out, ensure_ascii=False)
    assert pipe.serializable_report()['official_production_blockers']


def test_crlf_non_lab_guard_and_deterministic_output():
    text='Bệnh nhân cao 1m70. Không ghi nhận đau ngực.\r\n'
    pipe1=EndToEndPipeline(cfg()); pipe2=EndToEndPipeline(cfg())
    out1=pipe1.infer_document(text); out2=pipe2.infer_document(text)
    assert out1 == out2
    assert all(row['text'] != '1m70' for row in out1)
    assert out1[0]['text'] == 'đau ngực'
    assert text[out1[0]['position'][0]:out1[0]['position'][1]] == 'đau ngực'


def test_candidate_code_dedupe_and_icd_auxiliary_not_used():
    pipe=EndToEndPipeline(cfg())
    out=pipe.infer_document(FIXTURE_TEXT)
    med=[r for r in out if r['type']=='THUỐC'][0]
    assert med['candidates'] == list(dict.fromkeys(med['candidates']))
    diag=[r for r in out if r['type']=='CHẨN_ĐOÁN'][0]
    assert diag['candidates'] == []
    assert any('ICD-10' in b for b in pipe.serializable_report()['official_production_blockers'])


def test_missing_required_artifacts_fail_closed(tmp_path):
    c=cfg(); c['production']=True; c['ner']={'mock':False}; c['assertion']={'mock':True}; c['rxnorm']['strict']=False
    with pytest.raises(FileNotFoundError, match='NER checkpoint'):
        EndToEndPipeline(c)
    c=cfg(); c['production']=False; c['rxnorm']['kb_dir']=str(tmp_path/'missing'); c['rxnorm']['strict']=True
    pipe=EndToEndPipeline(c)
    with pytest.raises(FileNotFoundError, match='RxNorm KB missing'):
        pipe.infer_document('Người bệnh dùng metformin 500 mg.')


def test_low_vram_mock_stage_orchestration_and_cache_hits():
    c=cfg(); c['runtime']['low_vram_mode']=True
    pipe=EndToEndPipeline(c)
    text='Người bệnh dùng metformin 500 mg. Sau đó dùng metformin 500 mg.'
    out=pipe.infer_document(text)
    meds=[r for r in out if r['type']=='THUỐC']
    assert all(m['candidates'] == ['6809'] for m in meds)
    assert pipe.serializable_report()['candidate_cache_hits'] >= 0


def test_non_mock_ner_uses_model_backend_not_rule(monkeypatch):
    import src.pipeline.end_to_end as e2e
    c=cfg(); c['ner']={'mock':False,'model_path':'dummy'}; c['assertion']={'mock':True}; c['rxnorm']['retrieval_mode']='lexical_smoke'
    called={'load':0,'predict':0}
    def fake_rule(text): raise AssertionError('rule NER fallback used')
    def fake_load(self):
        called['load'] += 1; self._ner_model=object(); self._ner_tokenizer=object(); self.report['model_metadata']['ner']={'backend':'test','mock':False,'model_forward_calls':0,'chunks':0,'inference_seconds':0.0}
    def fake_predict(text, tokenizer, model, **kw):
        called['predict'] += 1; return [{'text':'đau ngực','type':'TRIỆU_CHỨNG','start':15,'end':23,'score':1.0}]
    monkeypatch.setattr(e2e, '_rule_ner', fake_rule)
    monkeypatch.setattr(e2e.EndToEndPipeline, '_load_ner', fake_load)
    monkeypatch.setattr(e2e, 'predict_with_model', fake_predict)
    out=e2e.EndToEndPipeline(c).infer_document('Không ghi nhận đau ngực.')
    assert called == {'load':1,'predict':1}
    assert out[0]['assertions'] == ['isNegated']


def test_non_mock_assertion_passes_model_tokenizer_thresholds(monkeypatch):
    import src.pipeline.end_to_end as e2e
    c=cfg(); c['ner']={'mock':True}; c['assertion']={'mock':False,'model_path':'assertion','thresholds_path':'thresholds.json'}
    seen={}
    def fake_load(self):
        self._assertion_model='MODEL'; self._assertion_tokenizer='TOK'; self._assertion_thresholds={'isNegated':0.7}; self.report['model_metadata']['assertion']={'backend':'test','mock':False,'model_forward_calls':0,'example_count':0,'inference_seconds':0.0}
    def fake_predict(text, entities, model=None, tokenizer=None, thresholds=None, **kw):
        seen.update({'model':model,'tokenizer':tokenizer,'thresholds':thresholds,'entities':len(entities)})
        return [{**e,'assertions':['isNegated']} for e in entities]
    monkeypatch.setattr(e2e.EndToEndPipeline, '_load_assertion', fake_load)
    monkeypatch.setattr(e2e, 'predict_assertions', fake_predict)
    out=e2e.EndToEndPipeline(c).infer_document('Không ghi nhận đau ngực.')
    assert seen == {'model':'MODEL','tokenizer':'TOK','thresholds':{'isNegated':0.7},'entities':1}
    assert out[0]['assertions'] == ['isNegated']


def _dense_pipeline_cfg(tmp_path):
    from src.data.kb_schema import read_jsonl
    from src.linking.dense import DenseAliasIndex, MockDenseEncoder, dense_expected_manifest, save_dense_index
    c=cfg(); c['rxnorm']['retrieval_mode']='bge_dense'; c['rxnorm']['dense_index_dir']=str(tmp_path/'dense'); c['rxnorm']['model_name']='test-bge'; c['rxnorm']['model_revision']='main'; c['rxnorm']['batch_size']=2; c['rxnorm']['max_length']=16
    kb=Path(c['rxnorm']['kb_dir']); paths=sorted(kb.glob('*.jsonl')); records=[]
    for p in paths: records.extend(read_jsonl(p))
    manifest=dense_expected_manifest({'model_name':'test-bge','model_revision':'main','include_unverified':False,'batch_size':2,'max_length':16}, paths, dimension=4, candidate_universe=1)
    save_dense_index(DenseAliasIndex.build(records, MockDenseEncoder(dim=4), False, manifest), Path(c['rxnorm']['dense_index_dir']), manifest)
    return c


def test_bge_dense_pipeline_loads_dense_encoder_and_does_not_call_lexical_search(tmp_path, monkeypatch):
    import src.pipeline.end_to_end as e2e
    from src.linking.dense import MockDenseEncoder
    from src.linking.retrieval import LexicalIndex
    c=_dense_pipeline_cfg(tmp_path); c['ner']={'mock':True}; c['assertion']={'mock':True}
    calls={'encode':0}
    class SpyBGE:
        def __init__(self, **kw): self.enc=MockDenseEncoder(dim=4)
        def encode(self, texts): calls['encode'] += 1; return self.enc.encode(texts)
    def bomb_search(self, *a, **k): raise AssertionError('BM25 search used in dense-only mode')
    monkeypatch.setattr(e2e, 'BGEM3Backend', SpyBGE)
    monkeypatch.setattr(LexicalIndex, 'search', bomb_search)
    pipe=e2e.EndToEndPipeline(c)
    out=pipe.infer_document('Người bệnh dùng metformin 500 mg.')
    assert [r for r in out if r['type']=='THUỐC'][0]['candidates'] == ['6809']
    meta=pipe.serializable_report()['model_metadata']['rxnorm']
    assert meta['backend'] == 'bge_dense' and meta['dense_query_encode_calls'] == 1 and meta['dense_search_calls'] == 1
    assert calls['encode'] == 1


def test_dense_missing_cache_fails_closed(tmp_path):
    c=cfg(); c['rxnorm']['retrieval_mode']='bge_dense'; c['rxnorm']['dense_index_dir']=str(tmp_path/'missing')
    pipe=EndToEndPipeline(c)
    with pytest.raises(FileNotFoundError):
        pipe.infer_document('Người bệnh dùng metformin 500 mg.')


def test_low_vram_stage_release_order_for_nonmock_backends(monkeypatch, tmp_path):
    import src.pipeline.end_to_end as e2e
    c=_dense_pipeline_cfg(tmp_path); c['runtime']['low_vram_mode']=True; c['ner']={'mock':False,'model_path':'ner'}; c['assertion']={'mock':False,'model_path':'assert'}
    monkeypatch.setattr(e2e.EndToEndPipeline, '_load_ner', lambda self: (setattr(self,'_ner_model',object()), setattr(self,'_ner_tokenizer',object()), self.report['model_metadata'].setdefault('ner',{'backend':'test','mock':False,'model_forward_calls':0,'chunks':0,'inference_seconds':0.0})))
    monkeypatch.setattr(e2e, 'predict_with_model', lambda text, tok, model, **kw: [{'text':'metformin 500 mg','type':'THUỐC','start':16,'end':32,'score':1.0}])
    monkeypatch.setattr(e2e.EndToEndPipeline, '_load_assertion', lambda self: (setattr(self,'_assertion_model','M'), setattr(self,'_assertion_tokenizer','T'), setattr(self,'_assertion_thresholds',{}), self.report['model_metadata'].setdefault('assertion',{'backend':'test','mock':False,'model_forward_calls':0,'example_count':0,'inference_seconds':0.0})))
    monkeypatch.setattr(e2e, 'predict_assertions', lambda text, ents, **kw: ents)
    from src.linking.dense import MockDenseEncoder
    monkeypatch.setattr(e2e, 'BGEM3Backend', lambda **kw: MockDenseEncoder(dim=4))
    pipe=e2e.EndToEndPipeline(c); pipe.infer_documents(['Người bệnh dùng metformin 500 mg.'])
    report=pipe.serializable_report()
    assert report['stage_release_events'] == ['ner','assertion','rxnorm']
    assert report['low_vram_mode_executed'] is True


def test_production_runtime_verified_requires_real_counters(tmp_path, monkeypatch):
    import src.pipeline.end_to_end as e2e
    c=_dense_pipeline_cfg(tmp_path); c['production']=True; c['ner']={'mock':False,'model_path':'ner'}; c['assertion']={'mock':False,'model_path':'assert'}; c['rxnorm']['strict']=True
    monkeypatch.setattr(e2e.EndToEndPipeline, '_load_ner', lambda self: (setattr(self,'_ner_model',object()), setattr(self,'_ner_tokenizer',object()), self.report['model_metadata'].setdefault('ner',{'backend':'test','mock':False,'model_forward_calls':0,'chunks':0,'inference_seconds':0.0})))
    monkeypatch.setattr(e2e, 'predict_with_model', lambda text, tok, model, **kw: [{'text':'metformin 500 mg','type':'THUỐC','start':16,'end':32,'score':1.0}])
    monkeypatch.setattr(e2e.EndToEndPipeline, '_load_assertion', lambda self: (setattr(self,'_assertion_model','M'), setattr(self,'_assertion_tokenizer','T'), setattr(self,'_assertion_thresholds',{}), self.report['model_metadata'].setdefault('assertion',{'backend':'test','mock':False,'model_forward_calls':0,'example_count':0,'inference_seconds':0.0})))
    monkeypatch.setattr(e2e, 'predict_assertions', lambda text, ents, **kw: ents)
    from src.linking.dense import MockDenseEncoder
    monkeypatch.setattr(e2e, 'BGEM3Backend', lambda **kw: MockDenseEncoder(dim=4))
    pipe=e2e.EndToEndPipeline(c); pipe.infer_document('Người bệnh dùng metformin 500 mg.')
    assert pipe.serializable_report()['production_runtime_verified'] is True
    c['ner']['mock']=True
    pipe=e2e.EndToEndPipeline(c); pipe.infer_document('Người bệnh dùng metformin 500 mg.')
    assert pipe.serializable_report()['production_runtime_verified'] is False


def test_drug_assertions_are_serialized_for_negated_and_historical_medications():
    pipe=EndToEndPipeline(cfg())
    out=pipe.infer_document('Không dùng metformin. Trước đây đã dùng amlodipine.')
    meds={r['text']:r for r in out if r['type']=='THUỐC'}
    assert meds['metformin']['assertions'] == ['isNegated']
    assert meds['amlodipine']['assertions'] == ['isHistorical']


def test_lab_entities_excluded_from_assertion_model_and_identity_preserved(monkeypatch):
    import src.pipeline.end_to_end as e2e
    c=cfg(); c['ner']={'mock':True}; c['assertion']={'mock':False,'model_path':'assert','thresholds_path':'thresholds'}
    seen=[]
    def fake_load(self):
        self._assertion_model='M'; self._assertion_tokenizer='T'; self._assertion_thresholds={}; self.report['model_metadata']['assertion']={'backend':'test','mock':False,'model_forward_calls':0,'example_count':0,'inference_seconds':0.0}
    def fake_predict(text, entities, **kw):
        seen.extend((e['text'], e['type'], tuple(e['position'])) for e in entities)
        return [{**e,'assertions':['isNegated']} for e in entities]
    monkeypatch.setattr(e2e.EndToEndPipeline, '_load_assertion', fake_load)
    monkeypatch.setattr(e2e, 'predict_assertions', fake_predict)
    text='Không ghi nhận đau ngực. Xét nghiệm HbA1c 8.1%.'
    out=e2e.EndToEndPipeline(c).infer_document(text)
    assert all(t not in {'TÊN_XÉT_NGHIỆM','KẾT_QUẢ_XÉT_NGHIỆM'} for _,t,_ in seen)
    assert ('đau ngực','TRIỆU_CHỨNG',(15,23)) in seen
    assert any(r['text']=='HbA1c' and 'assertions' not in r for r in out)
    assert all(text[r['position'][0]:r['position'][1]] == r['text'] for r in out)


def test_repeated_medications_encode_one_unique_dense_query(tmp_path, monkeypatch):
    import src.pipeline.end_to_end as e2e
    from src.linking.dense import MockDenseEncoder
    c=_dense_pipeline_cfg(tmp_path); c['ner']={'mock':False,'model_path':'ner'}; c['assertion']={'mock':True}
    text=' '.join(['metformin.']*10)
    spans=[]; start=0
    for _ in range(10):
        spans.append({'text':'metformin','type':'THUỐC','start':start,'end':start+9,'score':1.0}); start += len('metformin. ')
    monkeypatch.setattr(e2e.EndToEndPipeline, '_load_ner', lambda self: (setattr(self,'_ner_model',object()), setattr(self,'_ner_tokenizer',object()), self.report['model_metadata'].setdefault('ner',{'backend':'test','mock':False,'model_forward_calls':0,'chunks':0,'inference_seconds':0.0})))
    monkeypatch.setattr(e2e, 'predict_with_model', lambda *a, **k: spans)
    calls={'encode':0,'queries':[]}
    class SpyBGE:
        def __init__(self, **kw): self.enc=MockDenseEncoder(dim=4)
        def encode(self, texts): calls['encode'] += 1; calls['queries'].extend(texts); return self.enc.encode(texts)
    monkeypatch.setattr(e2e, 'BGEM3Backend', SpyBGE)
    pipe=e2e.EndToEndPipeline(c); out=pipe.infer_document(text); report=pipe.serializable_report(); meta=report['model_metadata']['rxnorm']
    assert len([r for r in out if r['type']=='THUỐC']) == 10
    assert calls['encode'] == 1 and calls['queries'] == ['metformin']
    assert meta['dense_query_encode_batches'] == 1 and meta['dense_query_vectors'] == 1 and meta['dense_search_calls'] == 1
    assert report['medication_entity_count'] == 10 and report['unique_medication_query_count'] == 1 and report['candidate_cache_hits'] >= 9


def test_strength_context_changes_dense_query_key_and_mapping(tmp_path, monkeypatch):
    import src.pipeline.end_to_end as e2e
    from src.linking.dense import MockDenseEncoder
    c=_dense_pipeline_cfg(tmp_path); c['ner']={'mock':False,'model_path':'ner'}; c['assertion']={'mock':True}
    text='Dùng metformin 500 mg đường uống. Dùng metformin 1000 mg đường uống.'
    spans=[{'text':'metformin','type':'THUỐC','start':5,'end':14,'score':1.0},{'text':'metformin','type':'THUỐC','start':39,'end':48,'score':1.0}]
    monkeypatch.setattr(e2e.EndToEndPipeline, '_load_ner', lambda self: (setattr(self,'_ner_model',object()), setattr(self,'_ner_tokenizer',object()), self.report['model_metadata'].setdefault('ner',{'backend':'test','mock':False,'model_forward_calls':0,'chunks':0,'inference_seconds':0.0})))
    monkeypatch.setattr(e2e, 'predict_with_model', lambda *a, **k: spans)
    calls=[]
    class SpyBGE:
        def __init__(self, **kw): self.enc=MockDenseEncoder(dim=4)
        def encode(self, texts): calls.extend(texts); return self.enc.encode(texts)
    monkeypatch.setattr(e2e, 'BGEM3Backend', SpyBGE)
    out=e2e.EndToEndPipeline(c).infer_document(text)
    assert calls == ['metformin 500 mg đường uống', 'metformin 1000 mg đường uống']
    assert [r['text'] for r in out if r['type']=='THUỐC'] == ['metformin','metformin']
    assert all(text[r['position'][0]:r['position'][1]] == r['text'] for r in out)
    assert all(r['candidates'] == ['6809'] for r in out if r['type']=='THUỐC')
