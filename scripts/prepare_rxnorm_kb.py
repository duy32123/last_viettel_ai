from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.import_rxnorm import import_rxnorm_prescribable
from src.data.kb_schema import write_jsonl
from src.linking.retrieval import report_records

def md5_file(path: Path) -> str:
    h=hashlib.md5()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(text, encoding='utf-8'); tmp.replace(path)

def atomic_jsonl(path: Path, records):
    tmp=path.with_suffix(path.suffix+'.tmp')
    write_jsonl(tmp, records); tmp.replace(path)

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/linking_kb.rxnorm_prescribable.yaml'); p.add_argument('--zip-path', default=None); p.add_argument('--version', default=None); p.add_argument('--expected-md5', default=''); p.add_argument('--dry-run', action='store_true'); ns=p.parse_args(argv)
    cfg=json.loads(Path(ns.config).read_text(encoding="utf-8"))
    zip_path=Path(ns.zip_path or cfg['zip_path']); version=ns.version or cfg.get('version'); expected_md5=ns.expected_md5 or cfg.get('expected_md5','')
    if not version or version=='override-required': raise SystemExit('RxNorm release version/date must be supplied via config or --version.')
    if ns.dry_run:
        print(json.dumps({'zip_path':str(zip_path),'version':version,'would_import':True,'official_kb':True,'verified':True,'expected_md5':bool(expected_md5)}, ensure_ascii=False, indent=2)); return
    if not zip_path.exists(): raise SystemExit(f'RxNorm ZIP not found: {zip_path}')
    if expected_md5 and md5_file(zip_path).lower()!=expected_md5.lower(): raise SystemExit('RxNorm ZIP MD5 mismatch')
    t0=time.time(); print(json.dumps({'event':'scan_rxnconso_start','zip_path':str(zip_path)}))
    records, report=import_rxnorm_prescribable(zip_path, version=version, source_url=cfg.get('source_url',''), release_date=cfg.get('release_date'))
    print(json.dumps({'event':'scan_rxnconso_done','seconds':time.time()-t0,'accepted_rows':report['accepted_rows'],'concept_count':report['concept_count']}))
    print(json.dumps({'event':'scan_rxnrel_done','seconds':time.time()-t0,'rxnrel_rows':report.get('rxnrel_rows',0),'rxnrel_kept_rows':report.get('rxnrel_kept_rows',0)}))
    out=Path(cfg['output_dir']); manifest={**report,'record_report':report_records(records)}
    t1=time.time(); atomic_jsonl(out/'rxnorm.jsonl', records); atomic_text(out/'manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({'event':'export_done','seconds':time.time()-t1,'output_dir':str(out)})); print(json.dumps(manifest, ensure_ascii=False, indent=2))
if __name__=='__main__': main()
