import json, re
from pathlib import Path
from src.data.balanced_ner_corpus import BalanceConfig, INVENTORY, apply_source_target_constraints, assert_balanced_gate, balanced_report, build_balanced_corpus, canonical_surface, generate_targeted_synthetic, source_target_allowed
from src.data.dataset_schema import VALID_TYPES

FAKE_PATTERNS=[re.compile(r"-\d{3}$"), re.compile(r"\btype \d{3}\b", re.I), re.compile(r"\bmức \d{3}\b", re.I)]


def test_source_target_constraints():
    assert source_target_allowed('DRUGCHEMICAL', 'THUỐC')
    assert source_target_allowed('DRUGCHEMICAL', 'TÊN_XÉT_NGHIỆM')
    assert not source_target_allowed('DRUGCHEMICAL', 'CHẨN_ĐOÁN')
    assert source_target_allowed('DISEASESYMTOM', 'TRIỆU_CHỨNG')
    assert source_target_allowed('DISEASESYMTOM', 'CHẨN_ĐOÁN')
    assert source_target_allowed('UNITCALIBRATOR', 'KẾT_QUẢ_XÉT_NGHIỆM')
    assert not source_target_allowed('DIAGNOSTICS', 'KẾT_QUẢ_XÉT_NGHIỆM')


def test_apply_source_target_constraints_drops_invalid_qwen_mapping():
    rec={'id':'q','text':'abc','entities':[{'id':'E1','start':0,'end':3,'text':'abc','type':'CHẨN_ĐOÁN','assertions':[],'candidates':[],'metadata':{'source_label':'DRUGCHEMICAL','weak_label_method':'qwen_two_pass'}}],'relations':[],'source':'silver','source_split':'train','license':'fixture','metadata':{}}
    assert apply_source_target_constraints(rec) is None


def test_canonical_surface_normalizes_numbers_case_and_punctuation():
    assert canonical_surface('HbA1c') == canonical_surface('HBA1C')
    assert canonical_surface('7.2 mmol/L') == canonical_surface('8.1 mmol/L')
    assert canonical_surface('glucose!') == canonical_surface('GLUCOSE')


def test_targeted_synthetic_offsets_and_real_inventory_diversity():
    cfg=BalanceConfig(min_entities_per_type=500, min_canonical_concepts_per_type=15, min_canonical_concepts_drug_diagnosis=20, max_top_canonical_share=0.07, audit_samples_per_type=2)
    rows=generate_targeted_synthetic(cfg)
    seen_types={e['type'] for r in rows for e in r['entities']}
    assert seen_types == VALID_TYPES
    assert any('\r\n' in r['text'] for r in rows)
    assert any('HbA1c' in r['text'] or 'HBA1C' in r['text'] for r in rows)
    assert any('đường uống' in r['text'] for r in rows)
    for r in rows:
        for e in r['entities']:
            assert r['source_split'] == 'train'
            assert 0 <= e['start'] < e['end'] <= len(r['text'])
            assert r['text'][e['start']:e['end']] == e['text']
            assert not any(p.search(e['text']) for p in FAKE_PATTERNS)
            assert e['metadata']['concept_id'] in {cid for vals in INVENTORY.values() for cid, _ in vals}
    report=balanced_report(rows)
    assert report['entity_count_by_type']['THUỐC'] == 500
    assert report['canonical_concepts']['THUỐC'] == len(INVENTORY['THUỐC'])
    assert report['canonical_concepts']['THUỐC'] < 100
    assert report['context_template_count']['THUỐC'] < 25
    assert_balanced_gate(report, cfg)


def test_result_entities_are_value_only_and_compatible_contexts():
    rows=generate_targeted_synthetic(BalanceConfig(min_entities_per_type=80))
    test_names=['INR','HbA1c','HBA1C','CRP','glucose','WBC','creatinine','AST','ALT','hemoglobin','Hb']
    result_rows=[r for r in rows if r['entities'][0]['type'] == 'KẾT_QUẢ_XÉT_NGHIỆM']
    assert result_rows
    for r in result_rows:
        e=r['entities'][0]; cid=e['metadata']['concept_id']
        assert r['text'][e['start']:e['end']] == e['text']
        assert not any(name == e['text'] or e['text'].startswith(name + ' ') for name in test_names)
        before=r['text'][:e['start']]; after=r['text'][e['end']:]
        assert any(name in before or name in after for name in test_names + ['cúm A','SARS-CoV-2'])
        assert 'INR INR' not in r['text']
        assert 'HbA1c INR' not in r['text']
        if cid == 'result_inr': assert 'INR' in before
        if cid == 'result_hba1c_pct': assert 'HbA1c' in before
        if cid == 'result_crp_mgl': assert 'CRP' in before
        if cid == 'result_glucose_mmol': assert 'glucose' in before or 'glucose' in after
        if cid == 'result_wbc_gl': assert 'WBC' in before


def test_diagnosis_uses_diagnosis_context_not_symptom_template():
    rows=generate_targeted_synthetic(BalanceConfig(min_entities_per_type=60))
    diagnosis_rows=[r for r in rows if r['entities'][0]['type'] == 'CHẨN_ĐOÁN']
    assert diagnosis_rows
    assert all(not r['text'].startswith('Triệu chứng:') for r in diagnosis_rows)
    assert any(r['text'].startswith('Chẩn đoán hiện tại:') or r['text'].startswith('Tiền sử bệnh:') for r in diagnosis_rows)



def test_build_balanced_corpus_train_only_and_gold_dev_independent(tmp_path):
    train_silver=tmp_path/'train.silver.jsonl'; train_syn=tmp_path/'train.jsonl'
    train_silver.write_text('', encoding='utf-8')
    train_syn.write_text(json.dumps({'id':'dev_leak','text':'dev text','entities':[],'relations':[],'source':'fixture','source_split':'dev','license':'fixture','metadata':{}}, ensure_ascii=False)+'\n', encoding='utf-8')
    output=tmp_path/'processed'/'train.v2.balanced.jsonl'; audit=tmp_path/'annotation'/'audit_v2.todo.jsonl'; report_path=tmp_path/'processed'/'balanced_v2_report.json'
    gold=tmp_path/'annotation'/'gold_dev.todo.jsonl'; gold.parent.mkdir(parents=True); gold.write_text('gold sentinel\n', encoding='utf-8')
    cfg=BalanceConfig(min_entities_per_type=60, min_canonical_concepts_per_type=15, min_canonical_concepts_drug_diagnosis=20, max_top_canonical_share=0.07, audit_samples_per_type=3)
    report=build_balanced_corpus(train_silver, train_syn, output, audit, report_path, cfg)
    rows=[json.loads(l) for l in output.read_text(encoding='utf-8').splitlines()]
    assert rows and all(r['source_split'] == 'train' for r in rows)
    assert 'dev_leak' not in {r['id'] for r in rows}
    assert gold.read_text(encoding='utf-8') == 'gold sentinel\n'
    audit_rows=[json.loads(l) for l in audit.read_text(encoding='utf-8').splitlines()]
    assert len(audit_rows) == 3 * len(VALID_TYPES)
    assert all(r['review_status'] == 'pending' and not r['metadata']['gold_evaluation'] for r in audit_rows)
    assert report['source_composition']['targeted_synthetic'] > 0


def test_balanced_gate_blocks_top_share_and_missing_type():
    cfg=BalanceConfig(min_entities_per_type=2, min_canonical_concepts_per_type=2, min_canonical_concepts_drug_diagnosis=2, max_top_canonical_share=0.51)
    bad={'entity_count_by_type':{'THUỐC':2},'canonical_concepts':{'THUỐC':1},'top_canonical_share_by_type':{'THUỐC':1.0},'class_imbalance_ratio':None,'duplicate_records':0,'invalid_offsets':0}
    try:
        assert_balanced_gate(bad, cfg)
    except ValueError as exc:
        assert 'minimum' in str(exc) or 'top canonical share' in str(exc)
    else:
        raise AssertionError('gate should fail')
