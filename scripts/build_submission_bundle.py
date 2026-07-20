from __future__ import annotations
import argparse, json, shutil, tempfile, hashlib, os
from pathlib import Path

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def copytree(src,dst,dry=False):
    files=sorted(p for p in Path(src).rglob('*') if p.is_file()) if Path(src).exists() else []
    if dry: return {'path':str(src),'file_count':len(files),'size_bytes':sum(p.stat().st_size for p in files)}
    shutil.copytree(src,dst,symlinks=False,ignore=shutil.ignore_patterns('optimizer.pt','trainer_state.json','*.log','cache','__pycache__'))
    return {'path':str(dst),'file_count':len(files),'size_bytes':sum(p.stat().st_size for p in files)}
def main(argv=None):
    p=argparse.ArgumentParser();
    for a in ['ner-model','assertion-model','rxnorm-kb','rxnorm-dense-index','bge-model','output']: p.add_argument('--'+a, required=True)
    p.add_argument('--dry-run', action='store_true')
    ns=p.parse_args(argv); out=Path(ns.output)
    mapping={'ner':ns.ner_model,'assertion':ns.assertion_model,'rxnorm':ns.rxnorm_kb,'dense_index':ns.rxnorm_dense_index,'bge':ns.bge_model}
    inventory={'dry_run':ns.dry_run,'components':{}}
    if ns.dry_run:
        for k,v in mapping.items(): inventory['components'][k]=copytree(v, out/k, dry=True)
        print(json.dumps(inventory, indent=2)); return
    staging=Path(tempfile.mkdtemp(prefix='submission_bundle_'))
    try:
        rel={'ner':'models/ner','assertion':'models/assertion','rxnorm':'kb/rxnorm','dense_index':'indexes/rxnorm_bge_m3','bge':'models/bge-m3'}
        for k,src in mapping.items(): inventory['components'][k]=copytree(src, staging/rel[k])
        manifest={'schema_version':1,'official_evaluation':False,'ner':{'path':rel['ner']},'assertion':{'path':rel['assertion'],'thresholds':rel['assertion']+'/thresholds.json'},'rxnorm':{'path':rel['rxnorm'],'candidate_universe':62099},'dense_index':{'path':rel['dense_index']},'bge':{'path':rel['bge'],'model_name':'BAAI/bge-m3','revision':'main'},'include_unverified':False,'reranker_enabled':False,'low_vram_mode':True,'icd10':{'production_blocker':'official ICD-10 KB missing; diagnosis candidates suppressed'},'expected_schema':{'concept_keys':['text','type','position','assertions','candidates']}}
        (staging/'champion_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        sums=[]
        for f in sorted(p for p in staging.rglob('*') if p.is_file()): sums.append(f'{sha256(f)}  {f.relative_to(staging)}')
        (staging/'SHA256SUMS.txt').write_text('\n'.join(sums)+'\n')
        inventory['total_size_bytes']=sum(p.stat().st_size for p in staging.rglob('*') if p.is_file())
        (staging/'artifact_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
        if out.exists(): shutil.rmtree(out)
        staging.replace(out)
        print(json.dumps({'output':str(out),'total_size_bytes':inventory['total_size_bytes']}, indent=2))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
if __name__=='__main__': main()
