from __future__ import annotations
import argparse, json, shutil, tempfile, hashlib, os, subprocess
from datetime import datetime, timezone
from pathlib import Path

EXCLUDES = {'.git','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','artifacts','outputs','checkpoints'}
EXCLUDE_PREFIXES = ('data/processed', '.cache', 'wandb', 'mlruns')
CODE_ITEMS = ['src','scripts','submission','configs','docs','run_pipeline.py','extract.py','dicts.py','README.md','pyproject.toml','setup.cfg','pytest.ini']
WEIGHTS = ('model.safetensors','pytorch_model.bin','tf_model.h5','model.onnx')
TOKENIZER_HINTS = ('tokenizer.json','tokenizer.model','vocab.json','sentencepiece.bpe.model','spiece.model')

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def _git(code_root: Path, args: list[str]) -> str:
    return subprocess.check_output(['git', *args], cwd=code_root, text=True, encoding="utf-8").strip()

def git_sha(code_root: Path) -> str:
    return _git(code_root, ['rev-parse','HEAD'])

def worktree_dirty(code_root: Path) -> bool:
    return bool(_git(code_root, ['status','--porcelain']))

def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve()); return True
    except ValueError:
        return False

def _iter_files(src: Path):
    if not src.exists(): return []
    if src.is_file(): return [src]
    files=[]
    for p in sorted(src.rglob('*')):
        rel=p.relative_to(src).as_posix()
        if any(part in EXCLUDES for part in p.parts) or any(rel.startswith(x) for x in EXCLUDE_PREFIXES):
            continue
        if p.is_symlink():
            target=p.resolve()
            if not _inside(target, src):
                raise ValueError(f'symlink escapes source tree: {p}')
            continue
        if p.is_file(): files.append(p)
    return files

def _component_stats(path: Path, bundle_rel: str | None=None):
    files=_iter_files(path)
    return {'source_path':str(path),'bundle_path':bundle_rel,'file_count':len(files),'size_bytes':sum(p.stat().st_size for p in files),'sha256':sha256(path) if path.exists() and path.is_file() else None}

def _has_symlink(src: Path) -> bool:
    if src.is_symlink(): return True
    if not src.exists() or not src.is_dir(): return False
    for p in src.rglob('*'):
        rel=p.relative_to(src).as_posix()
        if any(part in EXCLUDES for part in p.parts) or any(rel.startswith(x) for x in EXCLUDE_PREFIXES):
            continue
        if p.is_symlink(): return True
    return False

def _copy_path(src: Path, dst: Path, *, dry=False, allow_symlinks=False):
    if _has_symlink(src) and not allow_symlinks:
        raise ValueError(f'{src} contains symlinks; run scripts/materialize_hf_snapshot.py first or pass --materialize-symlinks')
    files=_iter_files(src)
    if dry:
        return {'path':str(src),'file_count':len(files),'size_bytes':sum(p.stat().st_size for p in files)}
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src,dst)
    elif src.exists():
        for f in files:
            rel=f.relative_to(src); out=dst/rel; out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f,out)
    return {'path':str(dst),'file_count':len(files),'size_bytes':sum(p.stat().st_size for p in files)}

def _validate_model_dir(path: Path, name: str):
    if not path.exists(): raise FileNotFoundError(f'missing {name}: {path}')
    files={p.name for p in _iter_files(path)}
    if 'config.json' not in files: raise FileNotFoundError(f'{name} missing config.json')
    if not any(w in files for w in WEIGHTS): raise FileNotFoundError(f'{name} missing supported weight file')
    if not any(t in files for t in TOKENIZER_HINTS): raise FileNotFoundError(f'{name} missing tokenizer file')

def _write_atomic(path: Path, text: str):
    tmp=path.with_name(path.name+'.tmp'); tmp.write_text(text, encoding='utf-8'); tmp.replace(path)

def _rxnorm_candidate_universe(path: Path) -> int:
    try:
        from src.data.kb_schema import read_jsonl
        codes=set()
        for f in Path(path).glob('*.jsonl'):
            for r in read_jsonl(f):
                if r.verified: codes.add(r.code)
        return len(codes) or 62099
    except Exception:
        return 62099

def _copy_code(code_root: Path, dst: Path, dry=False):
    total={'path':str(dst),'file_count':0,'size_bytes':0}
    items=list(CODE_ITEMS) + [p.name for p in sorted(code_root.glob('requirements*.txt'))]
    for item in dict.fromkeys(items):
        src=code_root/item
        if not src.exists(): continue
        stat=_copy_path(src, dst/item, dry=dry)
        total['file_count'] += stat['file_count']; total['size_bytes'] += stat['size_bytes']
    wrapper=dst/'run_submission.py'
    if not dry:
        wrapper.write_text("from submission.run import main\nraise SystemExit(main())\n", encoding='utf-8')
        total['file_count'] += 1; total['size_bytes'] += wrapper.stat().st_size
    return total

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--code-root', required=True)
    for a in ['ner-model','assertion-model','rxnorm-kb','rxnorm-dense-index','bge-model','output']:
        p.add_argument('--'+a, required=True)
    p.add_argument('--icd10-kb')
    p.add_argument('--dry-run', action='store_true'); p.add_argument('--allow-dirty', action='store_true'); p.add_argument('--overwrite', action='store_true'); p.add_argument('--materialize-symlinks', action='store_true')
    ns=p.parse_args(argv); code_root=Path(ns.code_root).resolve(); out=Path(ns.output).resolve()
    sha=git_sha(code_root)
    dirty=worktree_dirty(code_root)
    if dirty and not ns.allow_dirty:
        raise RuntimeError('code worktree is dirty; commit changes or pass --allow-dirty')
    rel={'ner':'models/ner','assertion':'models/assertion','rxnorm':'kb/rxnorm','dense_index':'indexes/rxnorm_bge_m3','bge':'models/bge-m3','code':'code'}
    mapping={'ner':Path(ns.ner_model),'assertion':Path(ns.assertion_model),'rxnorm':Path(ns.rxnorm_kb),'dense_index':Path(ns.rxnorm_dense_index),'bge':Path(ns.bge_model)}
    if ns.icd10_kb:
        rel['icd10']='kb/icd10'; mapping['icd10']=Path(ns.icd10_kb)
    inventory={'dry_run':ns.dry_run,'code_git_sha':sha,'created_at':datetime.now(timezone.utc).isoformat(),'components':{}}
    if ns.dry_run:
        inventory['dirty']=dirty
        inventory['components']['code']=_copy_code(code_root, out/rel['code'], dry=True); inventory['components']['code']['bundle_path']=rel['code']; inventory['components']['code'].pop('path', None)
        for k,v in mapping.items():
            stat=_component_stats(v, rel[k]); stat.pop('source_path', None); inventory['components'][k]=stat
        print(json.dumps(inventory, ensure_ascii=False, indent=2)); return
    if out.exists() and not ns.overwrite:
        raise FileExistsError(f'output exists (use --overwrite): {out}')
    for src in mapping.values():
        if _has_symlink(src) and not ns.materialize_symlinks:
            raise ValueError(f'{src} contains symlinks; run scripts/materialize_hf_snapshot.py first or pass --materialize-symlinks')
    _validate_model_dir(mapping['ner'], 'NER model')
    _validate_model_dir(mapping['assertion'], 'assertion model')
    _validate_model_dir(mapping['bge'], 'BGE model')
    staging=Path(tempfile.mkdtemp(prefix=out.name+'.staging.', dir=str(out.parent)))
    try:
        inventory['components']['code']=_copy_code(code_root, staging/rel['code'])
        inventory['components']['code']['bundle_path']=rel['code']; inventory['components']['code'].pop('path', None)
        for k,src in mapping.items():
            stat=_copy_path(src, staging/rel[k], allow_symlinks=ns.materialize_symlinks)
            inventory['components'][k]={'bundle_path':rel[k],'file_count':stat['file_count'],'size_bytes':stat['size_bytes']}
        def comp(name): return {'path':rel[name], **inventory['components'][name]}
        candidate_universe=_rxnorm_candidate_universe(mapping['rxnorm'])
        manifest={'schema_version':1,'code_git_sha':sha,'created_at':inventory['created_at'],'official_evaluation':False,'ner':comp('ner'),'assertion':{**comp('assertion'),'thresholds':rel['assertion']+'/thresholds.json'},'rxnorm':{**comp('rxnorm'),'candidate_universe':candidate_universe},'dense_index':comp('dense_index'),'bge':{**comp('bge'),'model_name':'BAAI/bge-m3','revision':'main'},'include_unverified':False,'reranker_enabled':False,'low_vram_mode':True,'icd10':(comp('icd10') if 'icd10' in mapping else {'production_blocker':'official ICD-10 KB missing; diagnosis candidates suppressed','required':False}),'expected_schema':{'concept_keys':['text','type','position','assertions','candidates']}}
        _write_atomic(staging/'champion_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        inventory['total_size_bytes']=sum(p.stat().st_size for p in staging.rglob('*') if p.is_file())
        _write_atomic(staging/'artifact_inventory.json', json.dumps(inventory, ensure_ascii=False, indent=2))
        sums=[]
        for f in sorted(p for p in staging.rglob('*') if p.is_file()):
            sums.append(f'{sha256(f)}  {f.relative_to(staging).as_posix()}')
        _write_atomic(staging/'SHA256SUMS.txt', '\n'.join(sums)+'\n')
        if out.exists():
            backup=out.with_name(out.name+'.old')
            if backup.exists(): shutil.rmtree(backup)
            out.rename(backup)
            try:
                staging.replace(out); shutil.rmtree(backup, ignore_errors=True)
            except Exception:
                if out.exists(): shutil.rmtree(out, ignore_errors=True)
                backup.rename(out); raise
        else:
            staging.replace(out)
        print(json.dumps({'output':str(out),'total_size_bytes':inventory['total_size_bytes'],'code_git_sha':sha}, ensure_ascii=False, indent=2))
    finally:
        if staging.exists(): shutil.rmtree(staging, ignore_errors=True)
if __name__=='__main__': main()
