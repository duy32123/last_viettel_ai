from __future__ import annotations
import argparse, json
from pathlib import Path
ALLOWED={'text','type','position','assertions','candidates'}
TYPES={'TRIỆU_CHỨNG','CHẨN_ĐOÁN','THUỐC','TÊN_XÉT_NGHIỆM','KẾT_QUẢ_XÉT_NGHIỆM'}
ASSERTIONS=['isNegated','isFamily','isHistorical']
def _sort_key(p):
    try: return (0,int(p.stem),p.name)
    except ValueError: return (1,p.name)
def load_verified_codes(kb_dir=None, *, terminology=None, official_required=False):
    verified=set()
    if kb_dir:
        from src.data.kb_schema import read_jsonl
        for p in Path(kb_dir).glob('*.jsonl'):
            for r in read_jsonl(p):
                if terminology and r.terminology != terminology: continue
                if r.verified and (not official_required or r.metadata.get('official_kb', False)): verified.add(r.code)
    return verified
def load_rxnorm_codes(rxnorm_kb=None):
    return load_verified_codes(rxnorm_kb, terminology='RxNorm', official_required=False)
def load_icd10_codes(icd10_kb=None):
    return load_verified_codes(icd10_kb, terminology='ICD-10', official_required=True)
def validate_file_pair(input_txt, output_json, rxnorm_codes=None, icd10_codes=None):
    inp=Path(input_txt); out=Path(output_json)
    if not out.exists(): raise FileNotFoundError(f'missing output: {out}')
    text=inp.read_text(encoding='utf-8'); rows=json.loads(out.read_text(encoding='utf-8'))
    if not isinstance(rows, list): raise ValueError('output JSON must be a list')
    prev=(-1,-1); seen=set(); total=0
    for row in rows:
        if not isinstance(row, dict): raise ValueError('concept must be object')
        if set(row)-ALLOWED: raise ValueError(f'unsupported keys: {set(row)-ALLOWED}')
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
            if row['type']=='THUỐC' and rxnorm_codes is not None and any(c not in rxnorm_codes for c in row['candidates']): raise ValueError('unverified or unknown RxCUI candidate')
            if row['type']=='CHẨN_ĐOÁN' and row['candidates']:
                if icd10_codes is None: raise ValueError('diagnosis candidates require official ICD KB')
                if any(c not in icd10_codes for c in row['candidates']): raise ValueError('unverified or unknown ICD-10 candidate')
        total += 1
    return {'valid':True,'concept_count':total}
def validate(input_dir, output_dir, expected_count=None, rxnorm_kb=None, icd10_kb=None):
    inputs=sorted(Path(input_dir).glob('*.txt'), key=_sort_key); outputs=sorted(Path(output_dir).glob('*.json'), key=_sort_key)
    if expected_count is not None and len(outputs)!=expected_count: raise ValueError('unexpected output count')
    if {p.stem for p in inputs}!={p.stem for p in outputs}: raise ValueError('input/output stem mismatch')
    extras={p.stem for p in outputs}-{p.stem for p in inputs}
    if extras: raise ValueError(f'stale extra output: {sorted(extras)}')
    rxnorm_codes=load_rxnorm_codes(rxnorm_kb) if rxnorm_kb else None
    icd10_codes=load_icd10_codes(icd10_kb) if icd10_kb else None
    total=0
    for inp in inputs:
        total += validate_file_pair(inp, Path(output_dir)/(inp.stem+'.json'), rxnorm_codes, icd10_codes)['concept_count']
    return {'valid':True,'input_count':len(inputs),'output_count':len(outputs),'concept_count':total,'validated_rxcui_count':len(rxnorm_codes or []),'validated_icd10_candidate_count':len(icd10_codes or [])}
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--expected-count',type=int); p.add_argument('--rxnorm-kb'); p.add_argument('--icd10-kb')
    ns=p.parse_args(argv); report=validate(ns.input_dir, ns.output_dir, ns.expected_count, ns.rxnorm_kb, ns.icd10_kb); print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__=='__main__': main()
