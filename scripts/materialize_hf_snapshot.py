from __future__ import annotations
import argparse, hashlib, json, shutil, tempfile, stat
from pathlib import Path

WEIGHTS={'model.safetensors','pytorch_model.bin','tf_model.h5','model.onnx'}
TOKENIZER={'tokenizer.json','tokenizer.model','vocab.json','sentencepiece.bpe.model','spiece.model'}

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def _validate(out: Path):
    files={p.name for p in out.rglob('*') if p.is_file()}
    if 'config.json' not in files: raise FileNotFoundError('materialized snapshot missing config.json')
    if not (files & WEIGHTS): raise FileNotFoundError('materialized snapshot missing supported model weight')
    if not (files & TOKENIZER): raise FileNotFoundError('materialized snapshot missing tokenizer file')

def materialize(source: Path, output: Path):
    source=source.resolve(); output=output.resolve()
    if not source.exists() or not source.is_dir(): raise FileNotFoundError(f'source snapshot missing: {source}')
    staging=Path(tempfile.mkdtemp(prefix=output.name+'.staging.', dir=str(output.parent)))
    try:
        inventory=[]
        for p in sorted(source.rglob('*')):
            rel=p.relative_to(source); dst=staging/rel
            if p.is_symlink():
                target=p.resolve(strict=False)
                if not target.exists(): raise FileNotFoundError(f'broken symlink: {p}')
                if target.is_dir(): raise ValueError(f'directory symlink not allowed: {p}')
                if not target.is_file(): raise ValueError(f'special symlink target not allowed: {p}')
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(target, dst)
            elif p.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            elif p.is_file():
                mode=p.stat().st_mode
                if not stat.S_ISREG(mode): raise ValueError(f'special file not allowed: {p}')
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p,dst)
            else:
                raise ValueError(f'special file not allowed: {p}')
        if any(p.is_symlink() for p in staging.rglob('*')): raise ValueError('materialized output contains symlink')
        _validate(staging)
        for f in sorted(x for x in staging.rglob('*') if x.is_file()):
            inventory.append({'path':f.relative_to(staging).as_posix(),'size_bytes':f.stat().st_size,'sha256':sha256(f)})
        (staging/'artifact_inventory.json').write_text(json.dumps({'file_count':len(inventory),'files':inventory}, indent=2), encoding='utf-8')
        sums=[f"{row['sha256']}  {row['path']}" for row in inventory]
        (staging/'SHA256SUMS.txt').write_text('\n'.join(sums)+'\n', encoding='utf-8')
        if output.exists(): raise FileExistsError(f'output exists: {output}')
        staging.replace(output)
        return {'output':str(output),'file_count':len(inventory),'size_bytes':sum(r['size_bytes'] for r in inventory)}
    finally:
        if staging.exists(): shutil.rmtree(staging, ignore_errors=True)

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--source', required=True); p.add_argument('--output', required=True)
    ns=p.parse_args(argv); print(json.dumps(materialize(Path(ns.source), Path(ns.output)), indent=2))
if __name__=='__main__': main()
