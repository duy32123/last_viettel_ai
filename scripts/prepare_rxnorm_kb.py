from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.import_rxnorm import import_rxnorm_prescribable
from src.data.kb_schema import write_jsonl
from src.linking.retrieval import report_records

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/linking_kb.rxnorm_prescribable.yaml'); p.add_argument('--zip-path', default=None); p.add_argument('--version', default=None); p.add_argument('--dry-run', action='store_true'); ns=p.parse_args(argv)
    cfg=json.loads(Path(ns.config).read_text())
    zip_path=Path(ns.zip_path or cfg['zip_path']); version=ns.version or cfg.get('version')
    if not version or version=='override-required': raise SystemExit('RxNorm release version/date must be supplied via config or --version.')
    if ns.dry_run:
        print(json.dumps({'zip_path':str(zip_path),'version':version,'would_import':True,'official_kb':True,'verified':True}, ensure_ascii=False, indent=2)); return
    records, report=import_rxnorm_prescribable(zip_path, version=version, source_url=cfg.get('source_url',''), release_date=cfg.get('release_date'))
    out=Path(cfg['output_dir']); write_jsonl(out/'rxnorm.jsonl', records)
    manifest={**report,'record_report':report_records(records)}
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
if __name__=='__main__': main()
