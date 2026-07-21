from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.kb_schema import read_jsonl
from src.linking.dense import BGEM3Backend, DenseAliasIndex, save_dense_index, dense_expected_manifest

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/linking.bge_m3_pilot.yaml'); p.add_argument('--dry-run', action='store_true'); ns=p.parse_args(); cfg=json.loads(Path(ns.config).read_text(encoding="utf-8"))
 paths=sorted(Path(cfg['kb_dir']).glob('*.jsonl')); records=[]
 for pth in paths: records.extend(read_jsonl(pth))
 include=bool(cfg.get('include_unverified',False)); universe=sum(1 for r in records if r.verified or include)
 base=dense_expected_manifest(cfg, paths, candidate_universe=universe)
 if ns.dry_run:
  print(json.dumps({'records':len(records),'candidate_universe':universe,'alias_count':sum(1+len(r.aliases) for r in records if r.verified or include),'manifest':base}, ensure_ascii=False, indent=2)); return
 encoder=BGEM3Backend(base['model_name'], int(base['batch_size']), int(base['max_length']))
 idx=DenseAliasIndex.build(records, encoder, include, base); dim=idx._dim(); manifest={**base,'dimension':dim}
 save_dense_index(idx, Path(cfg['dense_index_dir']), manifest); print(json.dumps(manifest, ensure_ascii=False, indent=2))
if __name__=='__main__': main()
