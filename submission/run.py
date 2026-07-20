from __future__ import annotations
import argparse, json, os, sys, random
from pathlib import Path
os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_pipeline_v2 import _atomic_json, _sort_key
from src.pipeline.end_to_end import EndToEndPipeline
from scripts.validate_submission import validate_file_pair, load_rxnorm_codes
from scripts.submission_preflight import main as preflight_main

def _valid_output(input_path, output_path, rxnorm_codes=None):
    try:
        validate_file_pair(input_path, output_path, rxnorm_codes); return True
    except Exception:
        return False

def _run_preflight(root: Path, report_path: Path | None):
    out=report_path or (root/'artifacts/submission_preflight_report.json')
    preflight_main(['--bundle-root',str(root),'--manifest',str(root/'champion_manifest.json'),'--output',str(out)])
    return out

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('input_dir'); p.add_argument('output_dir'); p.add_argument('--bundle-root',required=True); p.add_argument('--resume',action='store_true'); p.add_argument('--overwrite',action='store_true'); p.add_argument('--expected-count',type=int); p.add_argument('--batch-size-docs', type=int, default=8); p.add_argument('--preflight-report')
    ns=p.parse_args(argv); random.seed(13)
    root=Path(ns.bundle_root).resolve(); _run_preflight(root, Path(ns.preflight_report) if ns.preflight_report else None)
    manifest=json.loads((root/'champion_manifest.json').read_text(encoding='utf-8'))
    cfg={'submission_mode':True,'production':True,'runtime':{'device':'auto','low_vram_mode':True},'ner':{'model_path':str(root/manifest['ner']['path']),'mock':False},'assertion':{'model_path':str(root/manifest['assertion']['path']),'thresholds_path':str(root/manifest['assertion']['thresholds']),'mock':False},'rxnorm':{'kb_dir':str(root/manifest['rxnorm']['path']),'dense_index_dir':str(root/manifest['dense_index']['path']),'bge_model_path':str(root/manifest['bge']['path']),'retrieval_mode':'bge_dense','include_unverified':False,'strict':True,'use_reranker':False,'top_k':10},'icd10':{'require':False},'diagnostics':{'report_path':str(root/'artifacts/submission_run_report.json')},'official_evaluation':False}
    inp=Path(ns.input_dir); out=Path(ns.output_dir); out.mkdir(parents=True, exist_ok=True); paths=sorted(inp.glob('*.txt'), key=_sort_key)
    expected_stems={p.stem for p in paths}; extra=sorted(p.stem for p in out.glob('*.json') if p.stem not in expected_stems)
    if extra: raise SystemExit(f'stale extra output files: {extra}')
    rx_codes=load_rxnorm_codes(root/manifest['rxnorm']['path'])
    to_run=[]; skipped=0
    for path in paths:
        op=out/(path.stem+'.json')
        if ns.resume and not ns.overwrite and _valid_output(path, op, rx_codes): skipped += 1
        else: to_run.append(path)
    pipe=EndToEndPipeline(cfg); failed=[]; processed=0
    for start in range(0, len(to_run), max(1, ns.batch_size_docs)):
        batch=to_run[start:start+max(1, ns.batch_size_docs)]
        try:
            results=pipe.infer_documents([p.read_text(encoding='utf-8') for p in batch])
            for path,res in zip(batch, results): _atomic_json(out/(path.stem+'.json'), res)
            processed += len(batch)
        except Exception as e:
            failed.extend([p.stem for p in batch]); break
    report=pipe.serializable_report(); report.update({'documents_processed':processed,'documents_skipped':skipped,'documents_failed':len(failed),'failed_stems':failed,'offline_mode':True,'network_download_attempts':0,'champion_manifest_checksum':None})
    rpath=Path(cfg['diagnostics']['report_path']); rpath.parent.mkdir(parents=True, exist_ok=True); _atomic_json(rpath, report)
    if failed or (ns.expected_count is not None and len(list(out.glob('*.json'))) != ns.expected_count): return 1
    return 0
if __name__=='__main__': raise SystemExit(main())
