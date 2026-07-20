from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
ALLOWED={'text','type','position','assertions','candidates'}
TYPES={'TRIỆU_CHỨNG','CHẨN_ĐOÁN','THUỐC','TÊN_XÉT_NGHIỆM','KẾT_QUẢ_XÉT_NGHIỆM'}
ASSERTIONS=['isNegated','isFamily','isHistorical']
def _sort_key(p):
    try: return (0,int(p.stem),p.name)
    except ValueError: return (1,p.name)
def validate(input_dir, output_dir, expected_count=None, rxnorm_kb=None):
    inputs=sorted(Path(input_dir).glob('*.txt'), key=_sort_key); outputs=sorted(Path(output_dir).glob('*.json'), key=_sort_key)
    if expected_count is not None and len(outputs)!=expected_count: raise ValueError('unexpected output count')
    if {p.stem for p in inputs}!={p.stem for p in outputs}: raise ValueError('input/output stem mismatch')
    verified=set()
    if rxnorm_kb:
        from src.data.kb_schema import read_jsonl
        for p in Path(rxnorm_kb).glob('*.jsonl'):
            for r in read_jsonl(p):
                if r.verified: verified.add(r.code)
    total=0
    for inp in inputs:
        text=inp.read_text(encoding='utf-8'); rows=json.loads((Path(output_dir)/(inp.stem+'.json')).read_text(encoding='utf-8'))
        if not isinstance(rows, list): raise ValueError('output JSON must be a list')
        prev=(-1,-1); seen=set()
        for row in rows:
            if set(row)-ALLOWED: raise ValueError(f'unsupported keys: {set(row)-ALLOWED}')
            if 'relations' in row or 'score' in row or 'metadata' in row: raise ValueError('internal field leaked')
            if row.get('type') not in TYPES: raise ValueError('invalid entity type')
            s,e=row.get('position',[None,None])
            if not (isinstance(s,int) and isinstance(e,int) and 0 <= s < e <= len(text)): raise ValueError('invalid span')
            if text[s:e] != row.get('text'): raise ValueError('offset text mismatch')
            if (s,e) in seen: raise ValueError('duplicate boundary')
            seen.add((s,e))
            if (s,e) < prev: raise ValueError('non-deterministic sort order')
            prev=(s,e)
            if 'assertions' in row:
                if row['assertions'] != [a for a in ASSERTIONS if a in row['assertions']]: raise ValueError('assertion order invalid')
            if 'candidates' in row:
                if not all(isinstance(c,str) for c in row['candidates']): raise ValueError('candidate must be string')
                if len(row['candidates']) != len(set(row['candidates'])): raise ValueError('duplicate candidates')
                if row['type']=='THUỐC' and verified and any(c not in verified for c in row['candidates']): raise ValueError('unverified RxCUI candidate')
                if row['type']=='CHẨN_ĐOÁN' and row['candidates']: raise ValueError('diagnosis candidates require official ICD KB')
            total += 1
    return {'valid':True,'input_count':len(inputs),'output_count':len(outputs),'concept_count':total}
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--expected-count',type=int); p.add_argument('--rxnorm-kb')
    ns=p.parse_args(argv); report=validate(ns.input_dir, ns.output_dir, ns.expected_count, ns.rxnorm_kb); print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__=='__main__': main()
