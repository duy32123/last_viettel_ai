from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.end_to_end import EndToEndPipeline, load_config


def _sort_key(path: Path):
    try: return (0, int(path.stem), path.name)
    except ValueError: return (1, path.name)

def _atomic_json(path: Path, obj):
    tmp=path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('input_dir'); p.add_argument('output_dir'); p.add_argument('--config', default='configs/pipeline.end_to_end.yaml')
    ns=p.parse_args(argv)
    cfg=load_config(ns.config)
    inp=Path(ns.input_dir); out=Path(ns.output_dir); out.mkdir(parents=True, exist_ok=True)
    pipe=EndToEndPipeline(cfg)
    paths=sorted(inp.glob('*.txt'), key=_sort_key)
    texts=[path.read_text(encoding='utf-8') for path in paths]
    results=pipe.infer_documents(texts)
    for path,concepts in zip(paths, results):
        _atomic_json(out/(path.stem + '.json'), concepts)
    report_path=Path(cfg.get('diagnostics',{}).get('report_path','artifacts/pipeline_run_report.json'))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report_path, pipe.serializable_report())
    print(json.dumps({'documents': pipe.report['documents'], 'output_dir': str(out), 'report_path': str(report_path), 'official_evaluation': False}, ensure_ascii=False, indent=2))

if __name__=='__main__': main()
