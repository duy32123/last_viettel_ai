from pathlib import Path
import argparse, json, sys, shutil
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.kb_schema import read_jsonl, file_sha256
from src.linking.pipeline import validate_kb
from src.linking.retrieval import LexicalIndex, save_lexical_index


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--config', default='configs/linking.yaml')
    p.add_argument('--dry-run', action='store_true')
    ns=p.parse_args(argv)
    cfg=json.loads(Path(ns.config).read_text(encoding="utf-8"))
    kb=Path(cfg['kb_dir'])
    out=Path(cfg.get('index_dir','data/processed/linking_index'))
    paths=sorted(kb.glob('*.jsonl')) if kb.exists() else []
    if ns.dry_run:
        print(json.dumps({
            'dry_run': True,
            'backend_executed': False,
            'kb_dir': str(kb),
            'kb_dir_exists': kb.exists(),
            'kb_file_count': len(paths),
            'index_dir': str(out),
            'production': bool(cfg.get('production', False)),
            'include_unverified': bool(cfg.get('include_unverified', False)),
            'expected_artifacts': ['lexical_index.json','lexical_manifest.json','lexical_postings_summary.json'],
        }, ensure_ascii=False, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    records=[]
    for pth in paths:
        records.extend(read_jsonl(pth))
        shutil.copyfile(pth, out/pth.name)
    report=validate_kb(records, bool(cfg.get('production',False)))
    report['kb_files']=[str(p) for p in paths]
    report['normalized_kb_hash']={p.name:file_sha256(p) for p in paths}
    idx=LexicalIndex(records, include_unverified=bool(cfg.get('include_unverified',False)))
    report['lexical_index']=save_lexical_index(idx, out, paths, {'production':bool(cfg.get('production',False))})
    tmp=out/'index_manifest.json.tmp'
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(out/'index_manifest.json')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__=='__main__': main()
