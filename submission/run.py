from __future__ import annotations
import argparse, json, os, sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_pipeline_v2 import _atomic_json, _sort_key
from src.pipeline.end_to_end import EndToEndPipeline
from scripts.validate_submission import validate

def _valid_output(input_path, output_path):
    if not output_path.exists(): return False
    try: validate(input_path.parent, output_path.parent); return True
    except Exception: return False

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('input_dir'); p.add_argument('output_dir'); p.add_argument('--bundle-root',required=True); p.add_argument('--resume',action='store_true'); p.add_argument('--overwrite',action='store_true'); p.add_argument('--expected-count',type=int)
    ns=p.parse_args(argv); random.seed(13); os.environ.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
    root=Path(ns.bundle_root); manifest=json.loads((root/'champion_manifest.json').read_text(encoding='utf-8'))
    cfg={'submission_mode':True,'production':True,'runtime':{'device':'auto','low_vram_mode':True},'ner':{'model_path':str(root/manifest['ner']['path']),'mock':False},'assertion':{'model_path':str(root/manifest['assertion']['path']),'thresholds_path':str(root/manifest['assertion']['thresholds']),'mock':False},'rxnorm':{'kb_dir':str(root/manifest['rxnorm']['path']),'dense_index_dir':str(root/manifest['dense_index']['path']),'bge_model_path':str(root/manifest['bge']['path']),'retrieval_mode':'bge_dense','include_unverified':False,'strict':True,'use_reranker':False,'top_k':10},'icd10':{'require':False},'diagnostics':{'report_path':str(root/'artifacts/submission_run_report.json')},'official_evaluation':False}
    inp=Path(ns.input_dir); out=Path(ns.output_dir); out.mkdir(parents=True, exist_ok=True); paths=sorted(inp.glob('*.txt'), key=_sort_key)
    to_run=[]; skipped=0
    for path in paths:
        op=out/(path.stem+'.json')
        if ns.resume and not ns.overwrite and _valid_output(path, op): skipped += 1
        else: to_run.append(path)
    pipe=EndToEndPipeline(cfg); failed=[]
    try:
        results=pipe.infer_documents([p.read_text(encoding='utf-8') for p in to_run])
        for path,res in zip(to_run, results): _atomic_json(out/(path.stem+'.json'), res)
    except Exception as e:
        failed.append(str(e))
    report=pipe.serializable_report(); report.update({'documents_processed':len(to_run),'documents_skipped':skipped,'documents_failed':len(failed),'errors':failed,'offline_mode':True,'network_download_attempts':0})
    rpath=Path(cfg['diagnostics']['report_path']); rpath.parent.mkdir(parents=True, exist_ok=True); _atomic_json(rpath, report)
    if failed or (ns.expected_count is not None and len(list(out.glob('*.json'))) != ns.expected_count): return 1
    return 0
if __name__=='__main__': raise SystemExit(main())
