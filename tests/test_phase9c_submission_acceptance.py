from __future__ import annotations
import json
from pathlib import Path
import pytest
from scripts import build_submission_bundle, submission_preflight, materialize_hf_snapshot
from src.data.kb_schema import KBRecord, write_jsonl
from src.linking.dense import AliasEntry, DenseAliasIndex, save_dense_index, dense_expected_manifest


def _model_dir(path: Path, thresholds=False):
    path.mkdir(parents=True)
    (path/'config.json').write_text('{}', encoding='utf-8')
    (path/'model.safetensors').write_bytes(b'x')
    (path/'tokenizer.json').write_text('{}', encoding='utf-8')
    if thresholds:
        (path/'thresholds.json').write_text(json.dumps({'isNegated':0.5,'isFamily':0.5,'isHistorical':0.5}), encoding='utf-8')

def _artifacts(tmp_path: Path):
    ner=tmp_path/'ner'; assertion=tmp_path/'assertion'; bge=tmp_path/'bge'; kb=tmp_path/'kb'; dense=tmp_path/'dense'
    _model_dir(ner); _model_dir(assertion, True); _model_dir(bge)
    rec=KBRecord('6809','metformin',['metformin'],'RxNorm','2026-07-06','RxNorm',True,{'official_kb':True},'drug','en')
    write_jsonl(kb/'rxnorm.jsonl', [rec])
    manifest=dense_expected_manifest({'model_name':'BAAI/bge-m3','model_revision':'main','batch_size':16,'max_length':8192,'include_unverified':False}, [kb/'rxnorm.jsonl'], dimension=2, candidate_universe=1)
    idx=DenseAliasIndex([AliasEntry('metformin','metformin','6809','metformin','RxNorm',True,'RxNorm','2026-07-06')], [[1.0,0.0]], manifest)
    save_dense_index(idx, dense, manifest)
    return ner, assertion, kb, dense, bge

def test_tiny_builder_manifest_relative_and_preflight_pass(tmp_path):
    ner, assertion, kb, dense, bge = _artifacts(tmp_path)
    out=tmp_path/'bundle'
    build_submission_bundle.main(['--code-root','.', '--ner-model',str(ner),'--assertion-model',str(assertion),'--rxnorm-kb',str(kb),'--rxnorm-dense-index',str(dense),'--bge-model',str(bge),'--output',str(out),'--allow-dirty'])
    manifest=json.loads((out/'champion_manifest.json').read_text())
    for key in ['ner','assertion','rxnorm','dense_index','bge']:
        assert not Path(manifest[key]['path']).is_absolute()
        assert 'source_path' not in manifest[key]
        assert str(tmp_path) not in json.dumps(manifest[key])
    inventory=json.loads((out/'artifact_inventory.json').read_text())
    assert str(tmp_path) not in json.dumps(inventory)
    assert not any(p.is_symlink() for p in out.rglob('*'))
    report=tmp_path/'preflight.json'
    submission_preflight.main(['--bundle-root',str(out),'--manifest',str(out/'champion_manifest.json'),'--output',str(report)])
    submission_preflight.main(['--bundle-root',str(out),'--manifest',str(out/'champion_manifest.json'),'--output',str(tmp_path/'preflight2.json')])

def test_materialize_hf_snapshot_dereferences_and_broken_link_fails(tmp_path):
    blobs=tmp_path/'blobs'; snap=tmp_path/'snap'; blobs.mkdir(); snap.mkdir()
    (blobs/'model.safetensors').write_bytes(b'w'); (snap/'model.safetensors').symlink_to(blobs/'model.safetensors')
    (snap/'config.json').write_text('{}'); (snap/'tokenizer.json').write_text('{}')
    out=tmp_path/'mat'
    materialize_hf_snapshot.main(['--source',str(snap),'--output',str(out)])
    assert not any(p.is_symlink() for p in out.rglob('*'))
    broken=tmp_path/'broken'; broken.mkdir(); (broken/'config.json').write_text('{}'); (broken/'tokenizer.json').write_text('{}'); (broken/'model.safetensors').symlink_to(tmp_path/'nope')
    with pytest.raises(FileNotFoundError):
        materialize_hf_snapshot.main(['--source',str(broken),'--output',str(tmp_path/'bad')])

def test_builder_rejects_symlink_snapshot_with_guidance(tmp_path):
    ner, assertion, kb, dense, bge = _artifacts(tmp_path)
    linked=tmp_path/'linked_bge'; linked.mkdir(); (linked/'config.json').write_text('{}'); (linked/'tokenizer.json').write_text('{}'); (linked/'model.safetensors').symlink_to(bge/'model.safetensors')
    with pytest.raises(ValueError, match='materialize_hf_snapshot'):
        build_submission_bundle.main(['--code-root','.', '--ner-model',str(ner),'--assertion-model',str(assertion),'--rxnorm-kb',str(kb),'--rxnorm-dense-index',str(dense),'--bge-model',str(linked),'--output',str(tmp_path/'bundle'),'--allow-dirty'])

def test_dry_run_code_inventory_matches_real_code_inventory(tmp_path):
    ner, assertion, kb, dense, bge = _artifacts(tmp_path)
    import io, contextlib
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        build_submission_bundle.main(['--code-root','.', '--ner-model',str(ner),'--assertion-model',str(assertion),'--rxnorm-kb',str(kb),'--rxnorm-dense-index',str(dense),'--bge-model',str(bge),'--output',str(tmp_path/'dry'),'--dry-run','--allow-dirty'])
    dry=json.loads(buf.getvalue())
    out=tmp_path/'bundle'
    build_submission_bundle.main(['--code-root','.', '--ner-model',str(ner),'--assertion-model',str(assertion),'--rxnorm-kb',str(kb),'--rxnorm-dense-index',str(dense),'--bge-model',str(bge),'--output',str(out),'--allow-dirty'])
    inv=json.loads((out/'artifact_inventory.json').read_text())
    assert dry['components']['code']['file_count'] == inv['components']['code']['file_count'] - 1  # run_submission.py wrapper added only in real copy
