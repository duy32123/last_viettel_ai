import json
from pathlib import Path
from src.data.weak_label import classify_candidate, weak_label_records, build_weak_corpus


def rec(text, mention, label, rid='r'):
    s=text.index(mention); e=s+len(mention)
    return {'id':rid,'text':text,'source':'fixture','source_split':'train','license':'fixture','source_entities':[{'start':s,'end':e,'text':mention,'source_label':label}], 'metadata':{'upstream_id':rid}}


class MockQwenBackend:
    model_name='mock-qwen'
    def __init__(self, labels):
        self.labels=labels; self.calls=[]
    def classify_batch(self, candidates, prompt_version):
        self.calls.append((prompt_version, candidates))
        value=self.labels[prompt_version]
        return [value for _ in candidates]


def test_inr_and_hba1c_are_test_names_not_drugs_or_results():
    text='INR cao, HBA1C 7.2%, đang dùng Prozac và Zoloft.'
    for mention in ['INR','HBA1C']:
        s=text.index(mention)
        d=classify_candidate(text,'DRUGCHEMICAL',s,s+len(mention))
        assert d.target == 'TÊN_XÉT_NGHIỆM'
    s=text.index('7.2%')
    assert classify_candidate(text,'UNITCALIBRATOR',s,s+len('7.2%')).target == 'KẾT_QUẢ_XÉT_NGHIỆM'
    for drug in ['Prozac','Zoloft']:
        s=text.index(drug); d=classify_candidate(text,'DRUGCHEMICAL',s,s+len(drug))
        assert d.target == 'THUỐC' and d.confidence >= 0.9


def test_organisms_ignore_and_obesity_diagnosis():
    for term in ['vi khuẩn','ruồi giấm']:
        text=f'Phát hiện {term} trong mẫu.'; s=text.index(term)
        assert classify_candidate(text,'DISEASESYMTOM',s,s+len(term)).target == 'IGNORE'
    text='Bệnh nhân béo phì.'; s=text.index('béo phì')
    assert classify_candidate(text,'DISEASESYMTOM',s,s+len('béo phì')).target == 'CHẨN_ĐOÁN'


def test_unitcalibrator_context_result_rules():
    text='Mức glucose cao.'; s=text.index('cao')
    assert classify_candidate(text,'UNITCALIBRATOR',s,s+3).target == 'KẾT_QUẢ_XÉT_NGHIỆM'
    text='Bệnh nhân cao tuổi.'; s=text.index('cao')
    assert classify_candidate(text,'UNITCALIBRATOR',s,s+3).target == 'IGNORE'
    text='WBC 12 G/L'; s=text.index('12 G/L')
    assert classify_candidate(text,'UNITCALIBRATOR',s,s+6).target == 'KẾT_QUẢ_XÉT_NGHIỆM'
    text='HBA1C được kiểm tra'; s=text.index('HBA1C')
    assert classify_candidate(text,'UNITCALIBRATOR',s,s+5).target == 'TÊN_XÉT_NGHIỆM'


def test_unicode_crlf_exact_offsets_and_train_only(tmp_path):
    text='BN sốt\r\nGlucose 8.2 mmol/L và dùng Prozac.'
    rows=[rec(text,'sốt','DISEASESYMTOM','train1'), rec(text,'8.2 mmol/L','UNITCALIBRATOR','train2'), rec(text,'Prozac','DRUGCHEMICAL','train3')]
    accepted, todo, report=weak_label_records(rows)
    assert not todo
    assert {e['type'] for r in accepted for e in r['entities']} == {'TRIỆU_CHỨNG','KẾT_QUẢ_XÉT_NGHIỆM','THUỐC'}
    for r in accepted:
        for e in r['entities']:
            assert 0 <= e['start'] < e['end'] <= len(r['text'])
            assert r['text'][e['start']:e['end']] == e['text']
            assert e['metadata']['weak_label_method'].startswith('rule')
            assert e['metadata']['confidence'] >= 0.9
    train=tmp_path/'train.real.jsonl'; synth=tmp_path/'syn.jsonl'; out=tmp_path/'out'; ann=tmp_path/'ann'
    train.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
    synth.write_text('',encoding='utf-8')
    (ann/'gold_dev.todo.jsonl').parent.mkdir(parents=True)
    (ann/'gold_dev.todo.jsonl').write_text('sentinel gold\n', encoding='utf-8')
    (ann/'local_test.todo.jsonl').write_text('sentinel test\n', encoding='utf-8')
    rep=build_weak_corpus(train,synth,out,ann)
    assert (out/'train.silver.jsonl').exists()
    assert (ann/'train.silver.todo.jsonl').exists()
    assert (ann/'gold_dev.todo.jsonl').read_text(encoding='utf-8') == 'sentinel gold\n'
    assert (ann/'local_test.todo.jsonl').read_text(encoding='utf-8') == 'sentinel test\n'
    assert rep['duplicate_leakage'] == 'checked_train_only_no_dev_test_inputs'


def test_qwen_backend_two_pass_accepts_only_agreement(tmp_path):
    rows=[rec('Bệnh nhân có biểu hiện lạ.','biểu hiện lạ','DISEASESYMTOM','q1')]
    backend=MockQwenBackend({'phase5d-weak-label-v1-a':'TRIỆU_CHỨNG','phase5d-weak-label-v1-b':'TRIỆU_CHỨNG'})
    accepted, todo, report=weak_label_records(rows, qwen_enabled=True, qwen_backend=backend, qwen_cache=tmp_path/'qwen.json')
    assert len(backend.calls) == 2
    assert accepted and not todo
    ent=accepted[0]['entities'][0]
    assert ent['type'] == 'TRIỆU_CHỨNG'
    assert ent['metadata']['qwen_agreement'] is True
    assert accepted[0]['text'][ent['start']:ent['end']] == ent['text']
    assert report['qwen_agreement']['agree'] == 1


def test_qwen_disagreement_stays_pending():
    rows=[rec('Bệnh nhân có biểu hiện lạ.','biểu hiện lạ','DISEASESYMTOM','q2')]
    backend=MockQwenBackend({'phase5d-weak-label-v1-a':'TRIỆU_CHỨNG','phase5d-weak-label-v1-b':'CHẨN_ĐOÁN'})
    accepted, todo, report=weak_label_records(rows, qwen_enabled=True, qwen_backend=backend)
    assert not accepted and todo
    ent=todo[0]['proposed_entities'][0]
    assert ent['review_status'] == 'pending'
    assert ent['metadata']['qwen_agreement'] is False
    assert report['qwen_agreement']['disagree'] == 1


def test_ambiguous_goes_to_todo_and_invalid_offsets_rejected():
    rows=[rec('Dùng thuốc lạ.','thuốc lạ','DRUGCHEMICAL','amb')]
    bad={'id':'bad','text':'abc','source_entities':[{'start':2,'end':1,'text':'','source_label':'DRUGCHEMICAL'}]}
    accepted, todo, report=weak_label_records(rows+[bad])
    assert not accepted and todo
    assert todo[0]['review_status'] == 'pending'
    assert report['invalid'] == 1
