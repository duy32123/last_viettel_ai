import json
from pathlib import Path
from src.data.balanced_ner_corpus import BalanceConfig, apply_source_target_constraints, assert_balanced_gate, balanced_report, build_balanced_corpus, generate_targeted_synthetic, source_target_allowed
from src.data.dataset_schema import VALID_TYPES


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



def test_targeted_synthetic_offsets_and_surface_diversity():
    cfg=BalanceConfig(min_entities_per_type=25, min_unique_mentions_per_type=10, min_unique_mentions_drug_diagnosis=10, max_top_mention_share=0.2, audit_samples_per_type=2)
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
    report=balanced_report(rows)
    assert_balanced_gate(report, cfg)


def test_build_balanced_corpus_train_only_and_gold_dev_independent(tmp_path):
    train_silver=tmp_path/'train.silver.jsonl'; train_syn=tmp_path/'train.jsonl'
    train_silver.write_text('', encoding='utf-8')
    train_syn.write_text(json.dumps({'id':'dev_leak','text':'dev text','entities':[],'relations':[],'source':'fixture','source_split':'dev','license':'fixture','metadata':{}}, ensure_ascii=False)+'\n', encoding='utf-8')
    output=tmp_path/'processed'/'train.v2.balanced.jsonl'; audit=tmp_path/'annotation'/'audit_v2.todo.jsonl'; report_path=tmp_path/'processed'/'balanced_v2_report.json'
    gold=tmp_path/'annotation'/'gold_dev.todo.jsonl'; gold.parent.mkdir(parents=True); gold.write_text('gold sentinel\n', encoding='utf-8')
    cfg=BalanceConfig(min_entities_per_type=20, min_unique_mentions_per_type=8, min_unique_mentions_drug_diagnosis=8, max_top_mention_share=0.2, audit_samples_per_type=3)
    report=build_balanced_corpus(train_silver, train_syn, output, audit, report_path, cfg)
    rows=[json.loads(l) for l in output.read_text(encoding='utf-8').splitlines()]
    assert rows and all(r['source_split'] == 'train' for r in rows)
    assert 'dev_leak' not in {r['id'] for r in rows}
    assert gold.read_text(encoding='utf-8') == 'gold sentinel\n'
    audit_rows=[json.loads(l) for l in audit.read_text(encoding='utf-8').splitlines()]
    assert len(audit_rows) == 3 * len(VALID_TYPES)
    assert all(r['review_status'] == 'pending' and not r['metadata']['gold_evaluation'] for r in audit_rows)
    assert report['class_imbalance_ratio'] <= 3


def test_balanced_gate_blocks_top_share_and_missing_type():
    cfg=BalanceConfig(min_entities_per_type=2, min_unique_mentions_per_type=2, min_unique_mentions_drug_diagnosis=2, max_top_mention_share=0.51)
    bad={'entity_count_by_type':{'THUỐC':2},'unique_forms_by_type':{'THUỐC':1},'top_mention_share_by_type':{'THUỐC':1.0},'class_imbalance_ratio':None,'duplicate_records':0,'invalid_offsets':0}
    try:
        assert_balanced_gate(bad, cfg)
    except ValueError as exc:
        assert 'minimum' in str(exc) or 'top mention share' in str(exc)
    else:
        raise AssertionError('gate should fail')
