from __future__ import annotations
import argparse, json, hashlib, zipfile
from pathlib import Path
from scripts.validate_submission import validate, _sort_key

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--zip-path',required=True); p.add_argument('--expected-count',type=int); p.add_argument('--rxnorm-kb', required=True)
    ns=p.parse_args(argv); report=validate(ns.input_dir, ns.output_dir, ns.expected_count, ns.rxnorm_kb)
    out=Path(ns.zip_path); tmp=out.with_suffix(out.suffix+'.tmp')
    files=sorted(Path(ns.output_dir).glob('*.json'), key=_sort_key)
    stems='\n'.join(f.stem for f in files).encode('utf-8')
    report['output_stem_checksum']=hashlib.sha256(stems).hexdigest(); report['zip_members']=[f.name for f in files]
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if f.name.startswith('.') or '/' in f.name: raise ValueError(f'invalid zip member: {f.name}')
            info=zipfile.ZipInfo(f.name, date_time=(1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED; z.writestr(info, f.read_bytes())
    tmp.replace(out); pkg={'zip_path':str(out),'output_zip_sha256':sha256(out),**report}; (out.with_suffix(out.suffix+'.report.json')).write_text(json.dumps(pkg,indent=2),encoding='utf-8'); print(json.dumps(pkg, indent=2))
if __name__=='__main__': main()
