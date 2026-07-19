from __future__ import annotations
import argparse, os, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.import_rxnorm import file_sha256

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--url', default=os.environ.get('RXNORM_PRESCRIBABLE_URL','')); p.add_argument('--output', default='data/raw/rxnorm/RxNorm_full_prescribe_current.zip'); p.add_argument('--dry-run', action='store_true'); ns=p.parse_args(argv)
    if not ns.url: raise SystemExit('Set --url or RXNORM_PRESCRIBABLE_URL; credentials must be supplied by environment if required.')
    out=Path(ns.output)
    if ns.dry_run:
        print({'url':ns.url,'output':str(out),'would_download':True}); return
    out.parent.mkdir(parents=True, exist_ok=True)
    req=urllib.request.Request(ns.url)
    token=os.environ.get('RXNORM_AUTH_TOKEN')
    if token: req.add_header('Authorization', f'Bearer {token}')
    with urllib.request.urlopen(req) as r, out.open('wb') as fh: fh.write(r.read())
    print({'output':str(out),'sha256':file_sha256(out)})
if __name__=='__main__': main()
