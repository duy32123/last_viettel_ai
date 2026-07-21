from __future__ import annotations
import argparse, hashlib, os, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.import_rxnorm import file_sha256

def md5_file(path: Path) -> str:
    h=hashlib.md5()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def _copy_response(resp, fh, chunk_size=1024*1024):
    while True:
        chunk=resp.read(chunk_size)
        if not chunk: break
        fh.write(chunk)

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--url', default=os.environ.get('RXNORM_PRESCRIBABLE_URL','')); p.add_argument('--output', default='data/raw/rxnorm/RxNorm_full_prescribe_current.zip'); p.add_argument('--expected-md5', default=''); p.add_argument('--resume', action='store_true'); p.add_argument('--dry-run', action='store_true'); ns=p.parse_args(argv)
    if not ns.url: raise SystemExit('Set --url or RXNORM_PRESCRIBABLE_URL; credentials must be supplied by environment if required.')
    out=Path(ns.output); part=out.with_suffix(out.suffix+'.part')
    if ns.dry_run:
        print({'url':ns.url,'output':str(out),'would_download':True,'resume':bool(ns.resume),'expected_md5':bool(ns.expected_md5)}); return
    out.parent.mkdir(parents=True, exist_ok=True)
    req=urllib.request.Request(ns.url)
    token=os.environ.get('RXNORM_AUTH_TOKEN')
    if token: req.add_header('Authorization', 'Bearer '+token)
    mode='wb'
    if ns.resume and part.exists() and part.stat().st_size>0:
        req.add_header('Range', f'bytes={part.stat().st_size}-'); mode='ab'
    with urllib.request.urlopen(req) as resp, part.open(mode) as fh:
        _copy_response(resp, fh)
    if ns.expected_md5 and md5_file(part).lower()!=ns.expected_md5.lower():
        raise SystemExit('Downloaded RxNorm ZIP MD5 mismatch')
    part.replace(out)
    print({'output':str(out),'sha256':file_sha256(out),'md5':md5_file(out)})
if __name__=='__main__': main()
