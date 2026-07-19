from pathlib import Path
import argparse, json, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.kb_schema import read_jsonl
from src.linking.dense import BGEM3Backend, DenseAliasIndex, save_dense_index, kb_checksum

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/linking.bge_m3_pilot.yaml'); p.add_argument('--dry-run', action='store_true'); ns=p.parse_args(); cfg=json.loads(Path(ns.config).read_text())
 paths=list(Path(cfg['kb_dir']).glob('*.jsonl')); records=[]
 for pth in paths: records.extend(read_jsonl(pth))
 manifest={'model_name':cfg.get('model_name','BAAI/bge-m3'),'model_revision':cfg.get('model_revision','main'),'dtype':'fp16_cuda_or_fp32_cpu','normalization':'l2','kb_checksum':kb_checksum(paths) if paths else None,'batch_size':cfg.get('batch_size',16),'max_length':cfg.get('max_length',8192),'include_unverified':bool(cfg.get('include_unverified',False)),'official_evaluation':False}
 if ns.dry_run:
  print(json.dumps({'records':len(records),'alias_count':sum(1+len(r.aliases) for r in records if r.verified or cfg.get('include_unverified')),'manifest':manifest}, ensure_ascii=False, indent=2)); return
 encoder=BGEM3Backend(manifest['model_name'], int(manifest['batch_size']), int(manifest['max_length']))
 idx=DenseAliasIndex.build(records, encoder, bool(cfg.get('include_unverified',False)), manifest); manifest['dimension']=len(idx.vectors[0]) if idx.vectors else 0
 save_dense_index(idx, Path(cfg['dense_index_dir']), manifest); print(json.dumps(manifest, ensure_ascii=False, indent=2))
if __name__=='__main__': main()
