import json
import zipfile
from pathlib import Path
import pytest
from scripts.validate_submission import validate
from scripts import package_submission


def test_validate_submission_schema_and_tamper_fail(tmp_path):
    out=tmp_path/'out'; out.mkdir()
    src=Path('tests/fixtures/pipeline/expected')
    for p in src.glob('*.json'): (out/p.name).write_text(p.read_text(encoding='utf-8'), encoding='utf-8')
    report=validate('tests/fixtures/pipeline/input', out, rxnorm_kb='tests/fixtures/pipeline/rxnorm_kb')
    assert report['valid'] is True and report['output_count']==2
    rows=json.loads((out/'sample.json').read_text())
    rows[0]['relations']=[]
    (out/'sample.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    with pytest.raises(ValueError, match='unsupported keys'):
        validate('tests/fixtures/pipeline/input', out, rxnorm_kb='tests/fixtures/pipeline/rxnorm_kb')


def test_package_submission_zip_root_layout(tmp_path):
    out=tmp_path/'out'; out.mkdir()
    for p in Path('tests/fixtures/pipeline/expected').glob('*.json'):
        (out/p.name).write_text(p.read_text(encoding='utf-8'), encoding='utf-8')
    zpath=tmp_path/'submission.zip'
    package_submission.main(['--input-dir','tests/fixtures/pipeline/input','--output-dir',str(out),'--zip-path',str(zpath),'--expected-count','2','--rxnorm-kb','tests/fixtures/pipeline/rxnorm_kb'])
    with zipfile.ZipFile(zpath) as z:
        names=z.namelist()
    assert names == sorted(p.name for p in out.glob('*.json'))
    assert all('/' not in n and not n.startswith('.') for n in names)
    assert zpath.with_suffix('.zip.report.json').exists()


def test_submission_mode_forbids_mock_and_requires_local_bge():
    from src.pipeline.end_to_end import EndToEndPipeline, load_config
    c=load_config('configs/pipeline.end_to_end.smoke.yaml')
    c['submission_mode']=True; c['production']=True; c['ner']={'mock':True}; c['assertion']={'mock':False,'model_path':'x'}; c['rxnorm']['retrieval_mode']='bge_dense'; c['rxnorm']['strict']=False
    with pytest.raises(RuntimeError, match='mock backends'):
        EndToEndPipeline(c)
    c['ner']={'mock':False,'model_path':'x'}; c['assertion']={'mock':False,'model_path':'x'}
    with pytest.raises(FileNotFoundError, match='BGE_MODEL_PATH'):
        EndToEndPipeline(c)
