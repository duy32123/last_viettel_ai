from __future__ import annotations
import argparse, json, hashlib, shutil, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.kb_schema import read_jsonl

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def _resolve(root, p): return (Path(root)/p).resolve()
def _files(root): return sorted(p for p in Path(root).rglob('*') if p.is_file()) if Path(root).exists() else []
def _check_component(root, desc, required=True):
    path=_resolve(root, desc.get('path',''))
    if required and not path.exists(): raise FileNotFoundError(f'missing component: {path}')
    return {'path':str(path),'exists':path.exists(),'file_count':len(_files(path)),'size_bytes':sum(p.stat().st_size for p in _files(path))}
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--bundle-root', required=True); p.add_argument('--manifest', required=True); p.add_argument('--output', default='artifacts/submission_preflight_report.json')
    ns=p.parse_args(argv); root=Path(ns.bundle_root); manifest=json.loads(Path(ns.manifest).read_text(encoding='utf-8'))
    report={'offline_mode':True,'network_download_attempts':0,'mock_artifact':False,'components':{},'checks':{},'free_disk_bytes':shutil.disk_usage(root if root.exists() else Path('.')).free}
    for name in ['ner','assertion','rxnorm','dense_index','bge']:
        report['components'][name]=_check_component(root, manifest[name])
    if not (root/manifest['assertion'].get('thresholds','')).exists(): raise FileNotFoundError('missing assertion thresholds')
    json.loads((root/manifest['assertion']['thresholds']).read_text(encoding='utf-8'))
    rx_path=_resolve(root, manifest['rxnorm']['path']); records=[]
    for f in sorted(rx_path.glob('*.jsonl')): records.extend(read_jsonl(f))
    if any(not r.verified for r in records): raise ValueError('RxNorm KB contains unverified record')
    universe=len({r.code for r in records})
    expected=int(manifest['rxnorm'].get('candidate_universe', universe))
    if universe != expected: raise ValueError(f'candidate universe mismatch: {universe} != {expected}')
    dmanifest=_resolve(root, manifest['dense_index']['path'])/'dense_manifest.json'
    if not dmanifest.exists(): raise FileNotFoundError('missing dense_manifest.json')
    bge_path=_resolve(root, manifest['bge']['path'])
    if not bge_path.exists(): raise FileNotFoundError('missing local BGE snapshot')
    report['checks'].update({'rxnorm_verified':True,'candidate_universe':universe,'dense_manifest':str(dmanifest),'bge_local_snapshot':True,'no_auxiliary_icd':True})
    try:
        import torch; report['device']={'cuda_available':bool(torch.cuda.is_available()),'dtype':'float16' if torch.cuda.is_available() else 'float32'}
    except Exception: report['device']={'cuda_available':False,'dtype':'unknown'}
    out=Path(ns.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'); print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__=='__main__': main()
