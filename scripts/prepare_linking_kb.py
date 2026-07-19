from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.import_icd import import_icd_official
from src.data.import_rxnorm import import_rxnorm_rrf, import_rxnorm_csv
from src.data.import_hf_icd_aux import rows_from_hf, import_auxiliary_rows, make_pilot_examples
from src.data.kb_schema import dedupe_records, write_jsonl, file_sha256
from src.linking.pipeline import validate_kb

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/linking_kb.yaml'); ns=p.parse_args(); cfg=json.loads(Path(ns.config).read_text())
 out=Path(cfg.get('output_dir','data/processed/linking_kb')); out.mkdir(parents=True, exist_ok=True); all_records=[]; manifests=[]
 for src in cfg.get('sources',[]):
  path=Path(src['path'])
  if not path.exists(): manifests.append({**src,'status':'skipped_missing'}); continue
  if src['kind']=='icd_official': recs=import_icd_official(path, src.get('terminology','ICD-10'), src['version'], src['name'], bool(src.get('verified',True)), src.get('field_map'))
  elif src['kind']=='rxnorm_rrf': recs=import_rxnorm_rrf(path, src['version'], src['name'], bool(src.get('verified',True)))
  elif src['kind']=='rxnorm_csv': recs=import_rxnorm_csv(path, src['version'], src['name'], bool(src.get('verified',True)))
  elif src['kind']=='hf_icd_auxiliary':
   if not bool(src.get('enabled', False)):
    manifests.append({**src,'status':'skipped_disabled'}); continue
   rows=list(rows_from_hf(src.get('split'))); recs, aux_report=import_auxiliary_rows(rows, src.get('version','hf-pilot'), src.get('dataset','birgermoell/icd10-clinical-notes'))
   (out/'auxiliary_pilot_examples.json').write_text(json.dumps(make_pilot_examples(rows,[r.code for r in recs]), ensure_ascii=False, indent=2), encoding='utf-8')
   manifests.append({**src,'status':'loaded','report':aux_report}); all_records.extend(recs); continue
  else: manifests.append({**src,'status':'skipped_unsupported'}); continue
  all_records.extend(recs); manifests.append({**src,'status':'loaded','checksum_sha256': file_sha256(path) if path.is_file() else None})
 records=dedupe_records(all_records); report=validate_kb(records, bool(cfg.get('production',False)))
 by={'ICD-10':[],'ICD-10-CM':[],'RxNorm':[]}
 for r in records: by[r.terminology].append(r)
 for term,recs in by.items():
  if recs: write_jsonl(out/(term.lower().replace('-','')+'.jsonl'), recs)
 (out/'manifest.json').write_text(json.dumps({'sources':manifests,'report':report}, ensure_ascii=False, indent=2), encoding='utf-8')
 print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__=='__main__': main()
