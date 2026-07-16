import json
from src.data.assertion_generator import AssertionGenConfig, build_assertion_corpus
from src.models.assertion.inference import predict_assertions, merge_rule_model, serialize_assertions
from src.models.assertion.labels import ASSERTION_LABELS, to_vector, from_scores
from src.models.assertion.metrics import multilabel_metrics, tune_thresholds
from src.models.assertion.preprocess import make_examples
from src.models.assertion.rules import rule_assertions


def ent(text, mention, typ='CHẨN_ĐOÁN'):
    s=text.index(mention); return {'text':mention,'type':typ,'position':[s,s+len(mention)],'candidates':['C1']}

def labels_for(text, mention, typ='CHẨN_ĐOÁN'):
    return predict_assertions(text, [ent(text, mention, typ)])[0]['assertions']

def test_assertion_label_vectors_order_and_serialization():
    assert ASSERTION_LABELS == ['isNegated','isFamily','isHistorical']
    assert to_vector(['isHistorical','isNegated']) == [1,0,1]
    assert from_scores([0.7,0.1,0.8], {'isNegated':0.5,'isFamily':0.5,'isHistorical':0.5}) == ['isNegated','isHistorical']
    text='Không ghi nhận đau ngực.'; e=ent(text,'đau ngực','TRIỆU_CHỨNG')
    out=predict_assertions(text,[e])[0]
    assert serialize_assertions(out) == {'text':'đau ngực','type':'TRIỆU_CHỨNG','position':e['position'],'candidates':['C1'],'assertions':['isNegated']}

def test_rule_hard_suite_scope_combinations_and_order():
    assert labels_for('Không ghi nhận đau ngực.','đau ngực','TRIỆU_CHỨNG') == ['isNegated']
    text='Không sốt nhưng còn ho.'
    assert labels_for(text,'sốt','TRIỆU_CHỨNG') == ['isNegated']
    assert labels_for(text,'ho','TRIỆU_CHỨNG') == []
    assert labels_for('Mẹ bệnh nhân có đái tháo đường.','đái tháo đường') == ['isFamily']
    assert labels_for('Tiền sử gia đình có tăng huyết áp.','tăng huyết áp') == ['isFamily','isHistorical']
    text='Tiền sử hen phế quản. Hiện tại khó thở.'
    assert labels_for(text,'hen phế quản') == ['isHistorical']
    assert labels_for(text,'khó thở','TRIỆU_CHỨNG') == []
    assert labels_for('Theo dõi viêm phổi.','viêm phổi') == []
    text='Không những sốt mà còn ho.'
    assert labels_for(text,'sốt','TRIỆU_CHỨNG') == []
    assert labels_for(text,'ho','TRIỆU_CHỨNG') == []

def test_crlf_section_transition_no_diacritic_and_provenance():
    text='TIỀN SỬ\r\nHen phế quản.\r\nHIỆN TẠI\r\nKhó thở.'
    assert labels_for(text,'Hen phế quản') == ['isHistorical']
    assert labels_for(text,'Khó thở','TRIỆU_CHỨNG') == []
    res=rule_assertions('Khong ghi nhan sot.', ent('Khong ghi nhan sot.', 'sot', 'TRIỆU_CHỨNG'))
    assert res['labels'] == []
    res=rule_assertions('Không ghi nhận sốt.', ent('Không ghi nhận sốt.', 'sốt', 'TRIỆU_CHỨNG'))
    assert res['rule_hits'][0]['rule_id'] == 'neg_scope'
    assert res['rule_hits'][0]['cue_span'] == [0,14]

def test_preprocess_entity_centered_context_keeps_original_offsets():
    text='A'*400 + 'Không ghi nhận đau ngực.' + 'B'*400
    e=ent(text,'đau ngực','TRIỆU_CHỨNG'); e['assertions']=['isNegated']
    ex=make_examples([{'id':'r','text':text,'entities':[e]}], max_chars=80)[0]
    assert '<ENT_START> đau ngực <ENT_END>' in ex['input_text']
    assert '<TYPE_TRIỆU_CHỨNG>' in ex['input_text']
    assert text[e['position'][0]:e['position'][1]] == e['text']
    assert ex['labels'] == [1,0,0]

def test_hybrid_thresholds_metrics_and_merge():
    assert merge_rule_model(['isNegated'], [0.1,0.9,0.9], {'isNegated':0.5,'isFamily':0.5,'isHistorical':0.5}) == ['isNegated','isFamily','isHistorical']
    m=multilabel_metrics([['isNegated'], [], ['isFamily','isHistorical']], [['isNegated'], [], ['isFamily']])
    assert m['micro_precision'] == 1.0
    assert m['per_label']['isHistorical']['false_negative'] == 1
    th=tune_thresholds([[1,0,0],[0,1,0]], [[.9,.1,.1],[.2,.8,.1]])
    assert set(th) == set(ASSERTION_LABELS)

def test_assertion_corpus_builder_gate_leakage_offsets_and_audit(tmp_path):
    out=tmp_path/'assertion'; audit=tmp_path/'audit.jsonl'
    cfg=AssertionGenConfig(train_negated=20, train_family=20, train_historical=20, train_none=40, combo_min=5, dev_per_family=4, test_per_family=4, audit=10)
    report=build_assertion_corpus(out, audit, cfg)
    assert report['record_counts']['train'] > report['record_counts']['dev']
    assert report['label_positive_counts']['train']['isNegated'] >= 20
    assert report['combination_counts']['train']['NONE'] >= 40
    assert not report['leakage']
    assert all(v == 0 for v in report['duplicates'].values())
    train=[json.loads(l) for l in (out/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    for r in train:
        assert r['source_split'] == 'train'
        for e in r['entities']:
            s,en=e['position']; assert r['text'][s:en] == e['text']
    audit_rows=[json.loads(l) for l in audit.read_text(encoding='utf-8').splitlines()]
    assert len(audit_rows) == 10 and all(r['review_status'] == 'pending' for r in audit_rows)

def test_template_family_disjoint_in_report(tmp_path):
    out=tmp_path/'assertion'; audit=tmp_path/'audit.jsonl'
    report=build_assertion_corpus(out, audit, AssertionGenConfig(train_negated=10, train_family=10, train_historical=10, train_none=20, combo_min=3, dev_per_family=2, test_per_family=2, audit=2))
    fams={k:set(v) for k,v in report['template_families'].items()}
    assert fams['train'].isdisjoint(fams['dev'])
    assert fams['train'].isdisjoint(fams['test'])
    assert fams['dev'].isdisjoint(fams['test'])
