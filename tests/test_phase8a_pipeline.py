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
