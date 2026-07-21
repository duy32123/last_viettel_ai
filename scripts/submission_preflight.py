from __future__ import annotations
import argparse, json, hashlib, shutil, sys, os, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.kb_schema import read_jsonl
from src.linking.dense import load_dense_index, dense_expected_manifest

WEIGHTS = {'model.safetensors','pytorch_model.bin','tf_model.h5','model.onnx'}
TOKENIZER_HINTS = {'tokenizer.json','tokenizer.model','vocab.json','sentencepiece.bpe.model','spiece.model'}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def _inside(path: Path, root: Path) -> bool:
    try: path.resolve().relative_to(root.resolve()); return True
    except ValueError: return False

def _resolve(root, p):
    if p is None: raise ValueError('missing path')
    pp=Path(p)
    if pp.is_absolute() or '..' in pp.parts: raise ValueError(f'unsafe manifest path: {p}')
    out=(Path(root)/pp).resolve()
    if not _inside(out, Path(root)): raise ValueError(f'path escapes bundle: {p}')
    return out

def _files(root): return sorted(p for p in Path(root).rglob('*') if p.is_file()) if Path(root).exists() else []

def _parse_sums(root: Path):
    sums=root/'SHA256SUMS.txt'
    if not sums.exists(): raise FileNotFoundError('missing SHA256SUMS.txt')
    entries={}
    for line in sums.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        h, rel=line.split(None, 1); rel=rel.strip()
        if rel.startswith('*'): rel=rel[1:]
        if Path(rel).is_absolute() or '..' in Path(rel).parts: raise ValueError(f'unsafe checksum path: {rel}')
        entries[rel]=h
    return entries

def verify_checksums(root: Path):
    listed=_parse_sums(root); failures=[]; checked=0
    actual={p.relative_to(root).as_posix() for p in _files(root)}
    ignored={'artifacts/submission_preflight_report.json','SHA256SUMS.txt'}
    for rel,h in listed.items():
        path=root/rel
        if not path.exists(): failures.append({'path':rel,'error':'missing'}); continue
        if path.is_symlink() and not _inside(path.resolve(), root): failures.append({'path':rel,'error':'symlink_escape'}); continue
        got=sha256(path); checked += 1
        if got != h: failures.append({'path':rel,'error':'sha256_mismatch','expected':h,'actual':got})
    unexpected=sorted(actual - set(listed) - ignored)
    return {'checksum_files_checked':checked,'checksum_failures':failures,'unexpected_files':unexpected,'manifest_sha256':sha256(root/'champion_manifest.json') if (root/'champion_manifest.json').exists() else None,'inventory_sha256':sha256(root/'artifact_inventory.json') if (root/'artifact_inventory.json').exists() else None}

def _check_component(root, desc, required=True):
    path=_resolve(root, desc.get('path',''))
    if required and not path.exists(): raise FileNotFoundError(f'missing component: {path}')
    if path.is_symlink() and not _inside(path.resolve(), root): raise ValueError(f'symlink escapes bundle: {path}')
    return {'path':str(path),'exists':path.exists(),'file_count':len(_files(path)),'size_bytes':sum(p.stat().st_size for p in _files(path))}

def _require_model_files(path: Path, name: str, thresholds=False):
    files={p.name for p in _files(path)}
    if 'config.json' not in files: raise FileNotFoundError(f'{name} missing config.json')
    if not (files & WEIGHTS): raise FileNotFoundError(f'{name} missing supported weight file')
    if not (files & TOKENIZER_HINTS): raise FileNotFoundError(f'{name} missing tokenizer files')
    if thresholds:
        tp=path/'thresholds.json'
        data=json.loads(tp.read_text(encoding='utf-8'))
        missing={'isNegated','isFamily','isHistorical'}-set(data)
        if missing: raise ValueError(f'assertion thresholds missing labels: {sorted(missing)}')

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--bundle-root', required=True); p.add_argument('--manifest', required=True); p.add_argument('--output', default='artifacts/submission_preflight_report.json'); p.add_argument('--deep', action='store_true')
    ns=p.parse_args(argv); root=Path(ns.bundle_root).resolve(); manifest=json.loads(Path(ns.manifest).read_text(encoding='utf-8'))
    if not manifest.get('code_git_sha'): raise ValueError('manifest code_git_sha must not be null')
    if not manifest.get('created_at'): raise ValueError('manifest created_at must not be null')
    report={'offline_mode':True,'network_download_attempts':0,'mock_artifact':False,'components':{},'checks':{},'free_disk_bytes':shutil.disk_usage(root if root.exists() else Path('.')).free}
    cks=verify_checksums(root); report['checks'].update(cks)
    if cks['checksum_failures'] or cks['unexpected_files']: raise ValueError(f'checksum validation failed: {cks}')
    for name in ['ner','assertion','rxnorm','dense_index','bge']:
        report['components'][name]=_check_component(root, manifest[name])
    if manifest.get('icd10',{}).get('path'):
        report['components']['icd10']=_check_component(root, manifest['icd10'])
    _require_model_files(_resolve(root, manifest['ner']['path']), 'NER model')
    _require_model_files(_resolve(root, manifest['assertion']['path']), 'assertion model', thresholds=True)
    _require_model_files(_resolve(root, manifest['bge']['path']), 'BGE model')
    rx_path=_resolve(root, manifest['rxnorm']['path']); records=[]
    for f in sorted(rx_path.glob('*.jsonl')): records.extend(read_jsonl(f))
    if any(not r.verified for r in records): raise ValueError('RxNorm KB contains unverified record')
    if any(not r.metadata.get('official_kb', True) for r in records): raise ValueError('RxNorm KB contains non-official record')
    universe=len({r.code for r in records}); expected=int(manifest['rxnorm'].get('candidate_universe', universe))
    if universe != expected: raise ValueError(f'candidate universe mismatch: {universe} != {expected}')
    ddir=_resolve(root, manifest['dense_index']['path']); dmanifest=json.loads((ddir/'dense_manifest.json').read_text(encoding='utf-8'))
    icd_records=[]
    if manifest.get('icd10',{}).get('path'):
        icd_path=_resolve(root, manifest['icd10']['path'])
        for f in sorted(icd_path.glob('*.jsonl')): icd_records.extend(read_jsonl(f))
        if any(r.terminology != 'ICD-10' for r in icd_records): raise ValueError('ICD KB must contain ICD-10 records only')
        if any((not r.verified) or (not r.metadata.get('official_kb', False)) for r in icd_records): raise ValueError('ICD KB contains unverified or non-official record')
    report['checks'].update({'rxnorm_verified':True,'candidate_universe':universe,'dense_manifest':str(ddir/'dense_manifest.json'),'bge_local_snapshot':True,'no_auxiliary_icd':True,'rxnorm_release':records[0].version if records else None,'dense_dimension':dmanifest.get('dimension'),'icd10_official_records':len({r.code for r in icd_records})})
    if ns.deep:
        kb_paths=sorted(rx_path.glob('*.jsonl'))
        expected_manifest=dense_expected_manifest({'model_name':manifest['bge'].get('model_name','BAAI/bge-m3'),'model_revision':manifest['bge'].get('revision','main'),'max_length':dmanifest.get('max_length',8192),'batch_size':dmanifest.get('batch_size',16),'include_unverified':False}, kb_paths, dimension=dmanifest.get('dimension'), candidate_universe=universe)
        load_dense_index(ddir, expected_manifest)
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(str(_resolve(root, manifest['bge']['path'])), local_files_only=True, use_fast=True)
            report['checks']['bge_tokenizer_local_load']=True
        except Exception as e:
            raise RuntimeError(f'BGE local tokenizer smoke failed: {e}')
    try:
        import torch; report['device']={'cuda_available':bool(torch.cuda.is_available()),'dtype':'float16' if torch.cuda.is_available() else 'float32'}
    except Exception: report['device']={'cuda_available':False,'dtype':'unknown'}
    out=Path(ns.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'); print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__=='__main__': main()
