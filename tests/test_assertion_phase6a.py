import json
import pytest
from src.data.assertion_generator import AssertionGenConfig, build_assertion_corpus, canonical_hash
from src.models.assertion.inference import predict_assertions, merge_rule_model, serialize_assertions
from src.models.assertion.labels import ASSERTION_LABELS, to_vector, from_scores
from src.models.assertion.metrics import multilabel_metrics, tune_thresholds
from src.models.assertion.preprocess import make_examples, register_special_tokens, SPECIAL_TOKENS
from src.models.assertion.rules import rule_assertions
from scripts.train_assertion import training_args_kwargs, trainer_kwargs


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

def test_preprocess_entity_centered_context_and_special_tokens():
    text='A'*400 + 'Không ghi nhận đau ngực.' + 'B'*400
    e=ent(text,'đau ngực','TRIỆU_CHỨNG'); e['assertions']=['isNegated']
    ex=make_examples([{'id':'r','text':text,'entities':[e]}], max_chars=80)[0]
    assert '<ENT_START> đau ngực <ENT_END>' in ex['input_text']
    assert '<TYPE_TRIỆU_CHỨNG>' in ex['input_text']
    assert text[e['position'][0]:e['position'][1]] == e['text']
    assert ex['labels'] == [1,0,0]
    class Tok:
        def __init__(self): self.tokens=[]
        def add_special_tokens(self, spec): self.tokens.extend(spec['additional_special_tokens']); return len(spec['additional_special_tokens'])
        def __len__(self): return 100 + len(self.tokens)
    class Model:
        def __init__(self): self.size=None
        def resize_token_embeddings(self, n): self.size=n
    tok=Tok(); model=Model(); added=register_special_tokens(tok, model)
    assert added == len(SPECIAL_TOKENS) and model.size == len(tok)
    assert '<ENT_START>' in tok.tokens and '<TYPE_KẾT_QUẢ_XÉT_NGHIỆM>' in tok.tokens

def test_hybrid_thresholds_metrics_and_merge():
    assert merge_rule_model(['isNegated'], [0.1,0.9,0.9], {'isNegated':0.5,'isFamily':0.5,'isHistorical':0.5}) == ['isNegated','isFamily','isHistorical']
    m=multilabel_metrics([['isNegated'], [], ['isFamily','isHistorical']], [['isNegated'], [], ['isFamily']])
    assert m['micro_precision'] == 1.0 and m['per_label']['isHistorical']['false_negative'] == 1
    th=tune_thresholds([[1,0,0],[0,1,0],[0,0,1],[0,0,0]], [[.9,.1,.1],[.2,.8,.1],[.1,.2,.85],[.1,.1,.1]])
    assert set(th) == set(ASSERTION_LABELS) and all('threshold' in v and 'f1' in v for v in th.values())
    with pytest.raises(ValueError): tune_thresholds([[1,0,0],[1,1,0]], [[.9,.1,.1],[.8,.8,.1]])

def test_assertion_corpus_builder_gate_leakage_offsets_and_audit(tmp_path):
    out=tmp_path/'assertion'; audit=tmp_path/'audit.jsonl'
    cfg=AssertionGenConfig(train_examples=500, dev_examples=180, test_examples=180, min_split_label_pos=30, min_split_label_neg=30, min_none=35, combo_min=8, all_three_min=5, audit=10)
    report=build_assertion_corpus(out, audit, cfg)
    assert report['record_counts']['train'] > report['record_counts']['dev']
    for split in ['train','dev','test']:
        assert not any('True' in key for key in report['synthetic_gold_flags'][split])
        assert set(report['entity_example_counts'][split]) == {'TRIỆU_CHỨNG','CHẨN_ĐOÁN','TÊN_XÉT_NGHIỆM','KẾT_QUẢ_XÉT_NGHIỆM','THUỐC'}
        assert report['duplicates'][split] == 0
    assert not report['leakage']
    train=[json.loads(l) for l in (out/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    for r in train:
        assert r['source_split'] == 'train' and not r['metadata']['gold_evaluation'] and not r['metadata']['official_evaluation']
        for e in r['entities']:
            s,en=e['position']; assert r['text'][s:en] == e['text']
    audit_rows=[json.loads(l) for l in audit.read_text(encoding='utf-8').splitlines()]
    assert len(audit_rows) == 10 and all(r['review_status'] == 'pending' for r in audit_rows)

def test_semantic_hash_strips_numeric_prefix_and_split_families_disjoint(tmp_path):
    a={'text':'Lượt khám 1: Ghi nhận sốt.','entities':[{'text':'sốt','type':'TRIỆU_CHỨNG','position':[20,23],'assertions':[]}]}
    b={'text':'Phiên 99: Ghi nhận sốt.','entities':[{'text':'sốt','type':'TRIỆU_CHỨNG','position':[19,22],'assertions':[]}]}
    assert canonical_hash(a) == canonical_hash(b)
    report=build_assertion_corpus(tmp_path/'a', tmp_path/'audit.jsonl', AssertionGenConfig(train_examples=500, dev_examples=180, test_examples=180, min_split_label_pos=30, min_split_label_neg=30, min_none=35, combo_min=8, all_three_min=5, audit=2))
    fams={k:set(v) for k,v in report['template_families'].items()}
    assert fams['train'].isdisjoint(fams['dev']) and fams['train'].isdisjoint(fams['test']) and fams['dev'].isdisjoint(fams['test'])

def test_training_argument_compatibility_report_to_none_and_trainer_tokenizer_fallback():
    class ArgsEval:
        def __init__(self, output_dir=None, eval_strategy=None, **kwargs): self.kwargs=kwargs
    class ArgsOld:
        def __init__(self, output_dir=None, evaluation_strategy=None, **kwargs): self.kwargs=kwargs
    cfg={'output_dir':'o','learning_rate':1e-5,'per_device_train_batch_size':1,'per_device_eval_batch_size':1,'gradient_accumulation_steps':1,'warmup_ratio':0.1,'weight_decay':0.0,'save_total_limit':1,'seed':83,'eval_strategy':'epoch'}
    kw=training_args_kwargs(ArgsEval, cfg); assert kw['report_to'] == 'none' and 'eval_strategy' in kw
    kw=training_args_kwargs(ArgsOld, cfg); assert kw['report_to'] == 'none' and 'evaluation_strategy' in kw
    class TrainerNew:
        def __init__(self, processing_class=None, **kwargs): pass
    class TrainerOld:
        def __init__(self, tokenizer=None, **kwargs): pass
    assert 'processing_class' in trainer_kwargs(TrainerNew, 1,2,3,4,'tok',6,7,[]) and 'tokenizer' in trainer_kwargs(TrainerOld, 1,2,3,4,'tok',6,7,[])
