import json
import warnings
import pytest
from src.data.assertion_generator import AssertionGenConfig, build_assertion_corpus, canonical_hash
from src.models.assertion.inference import predict_assertions, merge_rule_model, serialize_assertions, load_thresholds, _model_scores
from src.models.assertion.labels import ASSERTION_LABELS, to_vector, from_scores
from src.models.assertion.metrics import multilabel_metrics, tune_thresholds
from src.models.assertion.preprocess import make_examples, register_special_tokens, SPECIAL_TOKENS, tokenize_examples, FloatMultilabelCollator, assert_float_multilabel_batch
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
    assert res['labels'] == ['isNegated']
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

def test_tokenizer_crop_keeps_markers_long_unicode_crlf_edges():
    pytest.importorskip("transformers")
    from transformers import BertTokenizerFast
    import tempfile
    vocab=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"] + SPECIAL_TOKENS + ["A","B","TIỀN","SỬ","đau","ngực","sốt","cuối","dòng"]
    with tempfile.TemporaryDirectory() as d:
        p=f"{d}/vocab.txt"; open(p,"w",encoding="utf-8").write("\n".join(vocab))
        tok=BertTokenizerFast(vocab_file=p, do_lower_case=False)
        register_special_tokens(tok)
        for text, mention in [
            ("đau ngực " + "A "*500, "đau ngực"),
            ("A "*300 + "TIỀN SỬ\r\nsốt\r\n" + "B "*300, "sốt"),
            ("A "*500 + "cuối dòng", "cuối dòng"),
        ]:
            e=ent(text, mention, "TRIỆU_CHỨNG"); e["assertions"]=["isNegated"]
            ex=make_examples([{"id":"r","text":text,"entities":[e]}], max_chars=160)[0]
            enc=tokenize_examples([ex], tok, max_length=64, padding=True)
            ids=enc["input_ids"][0]
            assert tok.convert_tokens_to_ids("<ENT_START>") in ids
            assert tok.convert_tokens_to_ids("<ENT_END>") in ids

def test_hybrid_thresholds_metrics_and_merge():
    text='Tiền sử gia đình có tăng huyết áp.'; e=ent(text,'tăng huyết áp')
    assert merge_rule_model(['isNegated'], [0.1,0.9,0.9], {'isNegated':0.5,'isFamily':0.5,'isHistorical':0.5}, text=text, entity=e) == ['isNegated','isFamily','isHistorical']
    m=multilabel_metrics([['isNegated'], [], ['isFamily','isHistorical']], [['isNegated'], [], ['isFamily']])
    assert m['micro_precision'] == 1.0 and m['per_label']['isHistorical']['false_negative'] == 1
    th=tune_thresholds([[1,0,0],[0,1,0],[0,0,1],[0,0,0]], [[.9,.1,.1],[.2,.8,.1],[.1,.2,.85],[.1,.1,.1]])
    assert set(th) == set(ASSERTION_LABELS) and all('threshold' in v and 'f1' in v for v in th.values())
    with pytest.raises(ValueError): tune_thresholds([[1,0,0],[1,1,0]], [[.9,.1,.1],[.8,.8,.1]])

def test_threshold_loading_flat_nested_and_rule_only_no_warning(tmp_path):
    flat=tmp_path/"flat.json"; flat.write_text(json.dumps({"isFamily":0.95}), encoding="utf-8")
    nested=tmp_path/"nested.json"; nested.write_text(json.dumps({"isFamily":{"threshold":0.9,"f1":1.0}}), encoding="utf-8")
    assert load_thresholds(flat)["isFamily"] == 0.95
    assert load_thresholds(nested)["isFamily"] == 0.9
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out=predict_assertions("Không ghi nhận đau ngực.", [ent("Không ghi nhận đau ngực.","đau ngực","TRIỆU_CHỨNG")])
    assert not record and out[0]["assertions"] == ["isNegated"]

def test_evidence_gated_hybrid_blocks_family_false_positives():
    thresholds={l:0.95 for l in ASSERTION_LABELS}
    fp1="Tiền sử bệnh nhân có đái tháo đường."
    out=predict_assertions(fp1, [ent(fp1,"đái tháo đường")], thresholds=thresholds, model_scores_override=[[0.0,0.99,0.99]])
    assert out[0]["assertions"] == ["isHistorical"]
    fp2="Hiện tại bệnh nhân có khó thở."
    out=predict_assertions(fp2, [ent(fp2,"khó thở","TRIỆU_CHỨNG")], thresholds=thresholds, model_scores_override=[[0.0,0.99,0.99]])
    assert out[0]["assertions"] == []
    ok="Gia đình ghi nhận khó thở."
    out=predict_assertions(ok, [ent(ok,"khó thở","TRIỆU_CHỨNG")], thresholds=thresholds, model_scores_override=[[0.0,0.99,0.0]])
    assert out[0]["assertions"] == ["isFamily"]

def test_phase6d_held_out_hard_suite_with_overconfident_model():
    cases=[
        ("Không ghi nhận đau ngực.","đau ngực","TRIỆU_CHỨNG",["isNegated"]),
        ("Không sốt nhưng còn ho.","ho","TRIỆU_CHỨNG",[]),
        ("Tiền sử bệnh nhân có đái tháo đường.","đái tháo đường","CHẨN_ĐOÁN",["isHistorical"]),
        ("Hiện tại bệnh nhân có khó thở.","khó thở","TRIỆU_CHỨNG",[]),
        ("Mẹ bệnh nhân có hen phế quản.","hen phế quản","CHẨN_ĐOÁN",["isFamily"]),
        ("Tiền sử gia đình có tăng huyết áp.","tăng huyết áp","CHẨN_ĐOÁN",["isFamily","isHistorical"]),
        ("TIỀN SỬ\r\nHen phế quản.\r\nHIỆN TẠI\r\nKhó thở.","Khó thở","TRIỆU_CHỨNG",[]),
        ("Ho so cu ghi viem phoi.","viem phoi","CHẨN_ĐOÁN",["isHistorical"]),
        ("Khong ghi nhan sot.","sot","TRIỆU_CHỨNG",["isNegated"]),
        ("Không những sốt mà còn ho.","sốt","TRIỆU_CHỨNG",[]),
    ]
    thresholds={l:0.95 for l in ASSERTION_LABELS}
    for text,mention,typ,expected in cases:
        out=predict_assertions(text,[ent(text,mention,typ)],thresholds=thresholds,model_scores_override=[[0.99,0.99,0.99]])
        assert out[0]["assertions"] == expected

def test_collator_rejects_long_labels_and_accepts_float32():
    torch=pytest.importorskip("torch")
    with pytest.raises(TypeError):
        assert_float_multilabel_batch({"labels":torch.tensor([[1,0,0]], dtype=torch.long)})
    assert_float_multilabel_batch({"labels":torch.tensor([[1,0,0]], dtype=torch.float32)})

def test_assertion_corpus_builder_gate_leakage_offsets_and_audit(tmp_path):
    out=tmp_path/'assertion'; audit=tmp_path/'audit.jsonl'
    cfg=AssertionGenConfig(train_examples=500, dev_examples=180, test_examples=180, min_split_label_pos=30, min_split_label_neg=30, min_none=35, combo_min=8, all_three_min=5, audit=10)
    report=build_assertion_corpus(out, audit, cfg)
    assert report['record_counts']['train'] > report['record_counts']['dev']
    for split in ['train','dev','test']:
        assert not any('True' in key for key in report['synthetic_gold_flags'][split])
        assert set(report['entity_example_counts'][split]) == {'TRIỆU_CHỨNG','CHẨN_ĐOÁN','TÊN_XÉT_NGHIỆM','KẾT_QUẢ_XÉT_NGHIỆM','THUỐC'}
        assert report['duplicates'][split] == 0
    assert report["semantic_template_overlap"] == {"train_vs_dev":0,"train_vs_test":0,"dev_vs_test":0}
    assert report["label_dtype"] == "float32"
    assert report["marker_truncation_count"] == 0
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

@pytest.mark.integration
def test_offline_trainer_step_save_reload_and_model_inference(tmp_path):
    torch=pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
    from datasets import Dataset
    vocab=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"] + SPECIAL_TOKENS + ["Không","ghi","nhận","đau","ngực","Mẹ","bệnh","nhân","có","sốt","Theo","dõi","Tiền","sử","gia","đình","."]
    vocab_path=tmp_path/"vocab.txt"; vocab_path.write_text("\n".join(vocab), encoding="utf-8")
    tok=BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
    cfg=BertConfig(vocab_size=len(tok)+len(SPECIAL_TOKENS), hidden_size=24, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, num_labels=3, problem_type="multi_label_classification")
    model=BertForSequenceClassification(cfg)
    register_special_tokens(tok, model)
    records=[
        {"id":"r1","text":"Không ghi nhận đau ngực.","entities":[{**ent("Không ghi nhận đau ngực.","đau ngực","TRIỆU_CHỨNG"),"assertions":["isNegated"]}]},
        {"id":"r2","text":"Mẹ bệnh nhân có sốt.","entities":[{**ent("Mẹ bệnh nhân có sốt.","sốt","TRIỆU_CHỨNG"),"assertions":["isFamily"]}]},
    ]
    examples=make_examples(records)
    def enc(batch):
        return tokenize_examples([{"input_text":t,"labels":l} for t,l in zip(batch["input_text"], batch["labels"])], tok, max_length=64, padding=False)
    ds=Dataset.from_list(examples).map(enc, batched=True, remove_columns=list(examples[0].keys()))
    args=TrainingArguments(output_dir=str(tmp_path/"out"), max_steps=1, per_device_train_batch_size=2, report_to="none", save_strategy="no")
    trainer=Trainer(model=model, args=args, train_dataset=ds, data_collator=FloatMultilabelCollator(tok))
    trainer.train()
    ckpt=tmp_path/"ckpt"; trainer.save_model(str(ckpt)); tok.save_pretrained(str(ckpt))
    (ckpt/"thresholds.json").write_text(json.dumps({l:0.0 for l in ASSERTION_LABELS}), encoding="utf-8")
    reloaded=BertForSequenceClassification.from_pretrained(str(ckpt))
    reloaded_tok=BertTokenizerFast.from_pretrained(str(ckpt))
    neutral_text="Theo dõi đau ngực."
    neutral_ent=ent(neutral_text,"đau ngực","TRIỆU_CHỨNG")
    scores=_model_scores(neutral_text, [neutral_ent], reloaded, reloaded_tok, batch_size=1)
    assert len(scores) == 1 and len(scores[0]) == 3
    assert all(0.0 <= float(p) <= 1.0 for p in scores[0])
    assert all(torch.isfinite(torch.tensor(scores[0])))
    out=predict_assertions(neutral_text, [neutral_ent], model=reloaded, tokenizer=reloaded_tok, thresholds={l:0.0 for l in ASSERTION_LABELS})
    assert out[0]["assertions"] == []
    cue_text="Tiền sử gia đình không ghi nhận đau ngực."
    cue_ent=ent(cue_text,"đau ngực","TRIỆU_CHỨNG")
    out=predict_assertions(cue_text, [cue_ent], model=reloaded, tokenizer=reloaded_tok, thresholds={l:0.0 for l in ASSERTION_LABELS})
    assert out[0]["assertions"] == ["isNegated","isFamily","isHistorical"]

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
