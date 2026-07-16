import json
from pathlib import Path
from src.data.weak_label import classify_candidate, weak_label_records, build_weak_corpus


def rec(text, mention, label, rid='r'):
    s=text.index(mention); e=s+len(mention)
    return {'id':rid,'text':text,'source':'fixture','source_split':'train','license':'fixture','source_entities':[{'start':s,'end':e,'text':mention,'source_label':label}], 'metadata':{'upstream_id':rid}}


def test_inr_not_drug_and_known_drugs_are_drugs():
    text='INR cao, đang dùng Prozac và Zoloft.'
    s=text.index('INR')
    assert classify_candidate(text,'DRUGCHEMICAL',s,s+3).target == 'IGNORE'
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
    rep=build_weak_corpus(train,synth,out,ann)
    assert (out/'train.silver.jsonl').exists()
    assert (ann/'train.silver.todo.jsonl').exists()
    assert rep['duplicate_leakage'] == 'checked_train_only_no_dev_test_inputs'


def test_ambiguous_goes_to_todo_and_invalid_offsets_rejected():
    rows=[rec('Dùng thuốc lạ.','thuốc lạ','DRUGCHEMICAL','amb')]
    bad={'id':'bad','text':'abc','source_entities':[{'start':2,'end':1,'text':'','source_label':'DRUGCHEMICAL'}]}
    accepted, todo, report=weak_label_records(rows+[bad])
    assert not accepted and todo
    assert todo[0]['review_status'] == 'pending'
    assert report['invalid'] == 1
