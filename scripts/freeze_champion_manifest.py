from __future__ import annotations
import argparse, json, subprocess, hashlib, datetime
from pathlib import Path


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def tree_files(root: Path): return sorted(p for p in root.rglob('*') if p.is_file()) if root.exists() else []
def component(root: Path):
    files=tree_files(root)
    return {'path':str(root),'file_count':len(files),'size_bytes':sum(p.stat().st_size for p in files),'checksums':{str(p.relative_to(root)):sha256(p) for p in files}}

def git_sha():
    try: return subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    except Exception: return None

def build_manifest(args):
    return {
        'schema_version':1,
        'code_git_sha':git_sha(),
        'created_at':datetime.datetime.utcnow().replace(microsecond=0).isoformat()+'Z',
        'official_evaluation':False,
        'ner':component(Path(args.ner_model)),
        'assertion':component(Path(args.assertion_model)),
        'assertion_thresholds':str(Path(args.assertion_model)/'thresholds.json'),
        'rxnorm':{**component(Path(args.rxnorm_kb)), 'release':args.rxnorm_release, 'candidate_universe':int(args.candidate_universe)},
        'dense_index':component(Path(args.rxnorm_dense_index)),
        'bge':{**component(Path(args.bge_model)), 'model_name':'BAAI/bge-m3','revision':args.bge_revision},
        'include_unverified':False,
        'reranker_enabled':False,
        'low_vram_mode':True,
        'icd10':(component(Path(args.icd10_kb)) if args.icd10_kb else {'production_blocker':'official ICD-10 KB missing; diagnosis candidates suppressed','auxiliary_icd_allowed_in_production':False}),
        'expected_schema':{'concept_keys':['text','type','position','assertions','candidates'],'entity_types':['TRIỆU_CHỨNG','CHẨN_ĐOÁN','THUỐC','TÊN_XÉT_NGHIỆM','KẾT_QUẢ_XÉT_NGHIỆM'],'assertion_order':['isNegated','isFamily','isHistorical']},
    }

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--ner-model',required=True); p.add_argument('--assertion-model',required=True); p.add_argument('--rxnorm-kb',required=True); p.add_argument('--icd10-kb'); p.add_argument('--rxnorm-dense-index',required=True); p.add_argument('--bge-model',required=True); p.add_argument('--output',default='configs/champion.phase9.json'); p.add_argument('--candidate-universe',default='62099'); p.add_argument('--rxnorm-release',default='2026-07-06'); p.add_argument('--bge-revision',default='main')
    ns=p.parse_args(argv); manifest=build_manifest(ns); out=Path(ns.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'); print(json.dumps({'output':str(out),'schema_version':1}, indent=2))
if __name__=='__main__': main()
