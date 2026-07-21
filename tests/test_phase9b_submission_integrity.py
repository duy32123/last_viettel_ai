from __future__ import annotations
import json, zipfile, subprocess
from pathlib import Path
import pytest
from scripts import submission_preflight, package_submission
from scripts.validate_submission import validate_file_pair
from src.data.kb_schema import KBRecord, write_jsonl
from src.linking.dense import AliasEntry, DenseAliasIndex, save_dense_index, dense_expected_manifest


def _sha(path: Path):
    from scripts.submission_preflight import sha256
    return sha256(path)

def _model_dir(path: Path, thresholds=False):
    path.mkdir(parents=True)
    (path/'config.json').write_text('{}', encoding='utf-8')
    (path/'model.safetensors').write_bytes(b'x')
    (path/'tokenizer.json').write_text('{}', encoding='utf-8')
    if thresholds:
        (path/'thresholds.json').write_text(json.dumps({'isNegated':0.5,'isFamily':0.5,'isHistorical':0.5}), encoding='utf-8')

def _bundle(tmp_path: Path):
    root=tmp_path/'bundle'
    _model_dir(root/'models/ner')
    _model_dir(root/'models/assertion', thresholds=True)
    _model_dir(root/'models/bge-m3')
    rec=KBRecord('6809','metformin',['metformin'],'RxNorm','2026-07-06','RxNorm',True,{'official_kb':True},'drug','en')
    write_jsonl(root/'kb/rxnorm/rxnorm.jsonl', [rec])
    manifest=dense_expected_manifest({'model_name':'BAAI/bge-m3','model_revision':'main','batch_size':16,'max_length':8192,'include_unverified':False}, [root/'kb/rxnorm/rxnorm.jsonl'], dimension=2, candidate_universe=1)
    idx=DenseAliasIndex([AliasEntry('metformin','metformin','6809','metformin','RxNorm',True,'RxNorm','2026-07-06')], [[1.0,0.0]], manifest)
    save_dense_index(idx, root/'indexes/rxnorm_bge_m3', manifest)
    champ={'schema_version':1,'code_git_sha':'abc','created_at':'2026-07-20T00:00:00+00:00','ner':{'path':'models/ner'},'assertion':{'path':'models/assertion','thresholds':'models/assertion/thresholds.json'},'rxnorm':{'path':'kb/rxnorm','candidate_universe':1},'dense_index':{'path':'indexes/rxnorm_bge_m3'},'bge':{'path':'models/bge-m3','model_name':'BAAI/bge-m3','revision':'main'},'include_unverified':False,'reranker_enabled':False}
    (root/'champion_manifest.json').write_text(json.dumps(champ), encoding='utf-8')
    inv={'components':{}}
    (root/'artifact_inventory.json').write_text(json.dumps(inv), encoding='utf-8')
    sums=[]
    for f in sorted(p for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt'):
        sums.append(f'{_sha(f)}  {f.relative_to(root).as_posix()}')
    (root/'SHA256SUMS.txt').write_text('\n'.join(sums)+'\n', encoding='utf-8')
    return root

def test_preflight_detects_tamper_missing_unexpected_and_nulls(tmp_path):
    root=_bundle(tmp_path)
    submission_preflight.main(['--bundle-root',str(root),'--manifest',str(root/'champion_manifest.json'),'--output',str(tmp_path/'r.json')])
    (root/'models/ner/config.json').write_text('{"tampered":true}', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum validation failed'):
        submission_preflight.main(['--bundle-root',str(root),'--manifest',str(root/'champion_manifest.json')])
    root=_bundle(tmp_path/'m')
    (root/'models/ner/config.json').unlink()
    with pytest.raises(ValueError, match='missing'):
        submission_preflight.main(['--bundle-root',str(root),'--manifest',str(root/'champion_manifest.json')])
    root=_bundle(tmp_path/'u')
    (root/'extra.bin').write_text('x', encoding="utf-8")
    with pytest.raises(ValueError, match='unexpected_files'):
        submission_preflight.main(['--bundle-root',str(root),'--manifest',str(root/'champion_manifest.json')])
    root=_bundle(tmp_path/'n')
    m=json.loads((root/'champion_manifest.json').read_text(encoding="utf-8")); m['code_git_sha']=None
    (root/'champion_manifest.json').write_text(json.dumps(m), encoding='utf-8')
    with pytest.raises(ValueError, match='code_git_sha'):
        submission_preflight.main(['--bundle-root',str(root),'--manifest',str(root/'champion_manifest.json')])

def test_preflight_rejects_path_escape_and_symlink_escape(tmp_path):
    root=_bundle(tmp_path)
    m=json.loads((root/'champion_manifest.json').read_text(encoding="utf-8")); m['ner']['path']='../escape'
    (root/'champion_manifest.json').write_text(json.dumps(m), encoding='utf-8')
    # update checksum so path gate is what fails
    lines=[]
    for f in sorted(p for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt'):
        lines.append(f'{_sha(f)}  {f.relative_to(root).as_posix()}')
    (root/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n', encoding="utf-8")
    with pytest.raises(ValueError, match='unsafe manifest path'):
        submission_preflight.main(['--bundle-root',str(root),'--manifest',str(root/'champion_manifest.json')])

def test_validate_file_pair_resume_semantics_and_corrupt(tmp_path):
    inp=Path('tests/fixtures/pipeline/input/sample.txt')
    good=tmp_path/'sample.json'; good.write_text(Path('tests/fixtures/pipeline/expected/sample.json').read_text(encoding='utf-8'), encoding='utf-8')
    rx={'6809'}
    assert validate_file_pair(inp, good, rx)['valid'] is True
    bad=tmp_path/'bad.json'; bad.write_text('{bad', encoding='utf-8')
    with pytest.raises(Exception):
        validate_file_pair(inp, bad, rx)

def test_package_requires_rxnorm_and_rejects_unknown_rxcui(tmp_path):
    out=tmp_path/'out'; out.mkdir()
    for p in Path('tests/fixtures/pipeline/expected').glob('*.json'):
        (out/p.name).write_text(p.read_text(encoding='utf-8'), encoding='utf-8')
    rows=json.loads((out/'sample.json').read_text(encoding="utf-8")); rows[2]['candidates']=['999999']
    (out/'sample.json').write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    with pytest.raises(ValueError, match='unknown RxCUI'):
        package_submission.main(['--input-dir','tests/fixtures/pipeline/input','--output-dir',str(out),'--zip-path',str(tmp_path/'x.zip'),'--rxnorm-kb','tests/fixtures/pipeline/rxnorm_kb'])

def test_package_deterministic_zip_numeric_order(tmp_path):
    inp=tmp_path/'inp'; out=tmp_path/'out'; inp.mkdir(); out.mkdir()
    for stem in ['10','2','a']:
        (inp/f'{stem}.txt').write_text('abc', encoding='utf-8')
        (out/f'{stem}.json').write_text('[]', encoding='utf-8')
    z=tmp_path/'s.zip'
    package_submission.main(['--input-dir',str(inp),'--output-dir',str(out),'--zip-path',str(z),'--expected-count','3','--rxnorm-kb','tests/fixtures/pipeline/rxnorm_kb'])
    with zipfile.ZipFile(z) as zz:
        assert zz.namelist()==['2.json','10.json','a.json']
