from __future__ import annotations
import json
from pathlib import Path
import pytest
from src.data.import_icd import import_icd_official
from src.data.kb_schema import write_jsonl
from src.pipeline.end_to_end import EndToEndPipeline
from scripts.validate_submission import validate, validate_file_pair, load_icd10_codes
from scripts import build_submission_bundle, submission_preflight, package_submission
from tests.test_phase9c_submission_acceptance import _artifacts, _clean_code_root


def test_import_icd_official_sets_verified_official_metadata(tmp_path):
    csv=tmp_path/'icd.csv'
    csv.write_text('code,title,aliases,parent,version,language,terminology\nI10,tăng huyết áp,tang huyet ap,,QD-4469-2020,vi,ICD-10\n', encoding='utf-8')
    rows=import_icd_official(csv, 'ICD-10', 'QD-4469-2020', 'QD4469', True)
    assert rows[0].verified is True
    assert rows[0].metadata['official_kb'] is True
    assert rows[0].metadata['source_kind'] == 'official_local'


def test_pipeline_uses_only_verified_official_icd_candidates(tmp_path):
    cfg=json.loads(Path('configs/pipeline.end_to_end.smoke.yaml').read_text(encoding='utf-8'))
    cfg['icd10']={'kb_dir':'tests/fixtures/pipeline/icd_official_kb','include_unverified':False,'require':False,'top_k':5}
    pipe=EndToEndPipeline(cfg)
    text=Path('tests/fixtures/pipeline/input/sample.txt').read_text(encoding='utf-8')
    rows=pipe.infer_document(text)
    diag=[r for r in rows if r['type']=='CHẨN_ĐOÁN'][0]
    assert diag['candidates'] == ['I10']
    bad=tmp_path/'bad_icd'; bad.mkdir()
    (bad/'icd10.jsonl').write_text('{"code":"I10","canonical_name":"tăng huyết áp","aliases":[],"terminology":"ICD-10","version":"x","semantic_type":"diagnosis","language":"vi","source":"aux","verified":true,"metadata":{"official_kb":false}}\n', encoding='utf-8')
    cfg['icd10']['kb_dir']=str(bad); cfg['production']=True
    with pytest.raises(ValueError, match='official_kb=true'):
        EndToEndPipeline(cfg).infer_document(text)


def test_validator_accepts_official_icd_and_rejects_auxiliary(tmp_path):
    inp=tmp_path/'in'; out=tmp_path/'out'; inp.mkdir(); out.mkdir()
    inp_txt=inp/'1.txt'; inp_txt.write_text('tăng huyết áp', encoding='utf-8')
    (out/'1.json').write_text(json.dumps([{'text':'tăng huyết áp','type':'CHẨN_ĐOÁN','position':[0,13],'candidates':['I10']}], ensure_ascii=False), encoding='utf-8')
    report=validate(inp, out, icd10_kb='tests/fixtures/pipeline/icd_official_kb')
    assert report['validated_icd10_candidate_count'] == 1
    with pytest.raises(ValueError, match='unknown ICD-10'):
        validate(inp, out, icd10_kb='tests/fixtures/pipeline/icd_aux_kb')


def test_bundle_manifest_preflight_and_package_support_icd10_kb(tmp_path):
    ner, assertion, kb, dense, bge = _artifacts(tmp_path)
    icd=tmp_path/'icd'; write_jsonl(icd/'icd10.jsonl', import_icd_official(Path('tests/fixtures/linking/icd10_fixture.csv'), 'ICD-10', 'TEST-2026', 'fixture', True))
    out=tmp_path/'bundle'
    build_submission_bundle.main(['--code-root',str(_clean_code_root(tmp_path)), '--ner-model',str(ner),'--assertion-model',str(assertion),'--rxnorm-kb',str(kb),'--icd10-kb',str(icd),'--rxnorm-dense-index',str(dense),'--bge-model',str(bge),'--output',str(out)])
    manifest=json.loads((out/'champion_manifest.json').read_text())
    assert manifest['icd10']['path'] == 'kb/icd10'
    submission_preflight.main(['--bundle-root',str(out),'--manifest',str(out/'champion_manifest.json'),'--output',str(tmp_path/'pf.json')])
    output=tmp_path/'output'; output.mkdir(); input_dir=tmp_path/'input'; input_dir.mkdir()
    input_dir.joinpath('1.txt').write_text('tăng huyết áp', encoding='utf-8')
    output.joinpath('1.json').write_text(json.dumps([{'text':'tăng huyết áp','type':'CHẨN_ĐOÁN','position':[0,13],'candidates':['I10']}], ensure_ascii=False), encoding='utf-8')
    package_submission.main(['--input-dir',str(input_dir),'--output-dir',str(output),'--zip-path',str(tmp_path/'s.zip'),'--rxnorm-kb',str(out/'kb/rxnorm'),'--icd10-kb',str(out/'kb/icd10')])
