from __future__ import annotations
import argparse, hashlib, json, random, sys
from collections import defaultdict, Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.kb_schema import read_jsonl

STRATA={
    'ingredient': {'IN','PIN','MIN'},
    'brand': {'BN'},
    'product': {'SCD','SBD','SCDC','SBDC','SCDF','SBDF'},
    'pack_form': {'GPCK','BPCK','DF'},
}

def split_code(code, seed=13):
    h=int(hashlib.sha256(f'{seed}:{code}'.encode()).hexdigest(),16)%100
    return 'train' if h<80 else ('dev' if h<90 else 'test')

def stratum(rec):
    tty=rec.metadata.get('TTY')
    for name,ttys in STRATA.items():
        if tty in ttys: return name
    return rec.metadata.get('specificity') or 'other'

def deterministic_sample(records, max_codes=800, seed=13):
    by=defaultdict(list)
    for rec in records: by[stratum(rec)].append(rec)
    selected=[]; rng=random.Random(seed); remaining=max_codes
    keys=sorted(by)
    quota=max(1, max_codes//max(1,len(keys)))
    for key in keys:
        rows=sorted(by[key], key=lambda r:r.code); rng.shuffle(rows)
        take=min(len(rows), quota, remaining); selected.extend(rows[:take]); remaining-=take
    if remaining>0:
        chosen={r.code for r in selected}; rest=[r for rows in by.values() for r in rows if r.code not in chosen]; rest=sorted(rest,key=lambda r:r.code); rng.shuffle(rest); selected.extend(rest[:remaining])
    return sorted(selected, key=lambda r:r.code)

def context_for(rec, examples_per_code=4):
    tty=rec.metadata.get('TTY'); name=rec.canonical_name; rows=[
        {'task':'exact_alias_diagnostic','mention':name,'context':name,'positive_code':rec.code,'language':'en','tty':tty,'specificity':stratum(rec)},
        {'task':'normalized_medication','mention':f'thuốc {name}','context':f'đang sử dụng thuốc {name} mỗi ngày','positive_code':rec.code,'language':'vi','tty':tty,'specificity':stratum(rec)},
        {'task':'brand_or_ingredient','mention':name,'context':f'bệnh nhân dùng {name}','positive_code':rec.code,'language':'vi','tty':tty,'specificity':stratum(rec)},
        {'task':'dose_form_context','mention':name,'context':f'{name} 500 mg đường uống','positive_code':rec.code,'language':'vi','tty':tty,'specificity':stratum(rec)},
        {'task':'noisy_vietnamese_mention','mention':name.replace('-', ' '),'context':f'bn dang dung {name.replace("-", " ")} moi ngay','positive_code':rec.code,'language':'vi_no_diacritic','tty':tty,'specificity':stratum(rec)},
    ]
    return rows[:max(1, examples_per_code)]

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--kb-dir', default='data/processed/linking_kb_rxnorm'); p.add_argument('--output', default='data/processed/linking_kb_rxnorm/rxnorm_pilot_examples.json'); p.add_argument('--max-codes', type=int, default=800); p.add_argument('--seed', type=int, default=13); p.add_argument('--examples-per-code', type=int, default=4); p.add_argument('--dry-run', action='store_true'); ns=p.parse_args(argv)
    records=[]
    for path in Path(ns.kb_dir).glob('*.jsonl'): records.extend(read_jsonl(path))
    selected=deterministic_sample(records, ns.max_codes, ns.seed)
    splits={'train':[],'dev':[],'test':[]}; code_split={}
    for rec in selected:
        sp=split_code(rec.code, ns.seed); code_split[rec.code]=sp
        for i,ex in enumerate(context_for(rec, ns.examples_per_code)):
            splits[sp].append({'id':f'{rec.code}:{i}', **ex, 'official_evaluation':False, 'synthetic_pilot':True})
    split_sets={sp:{e['positive_code'] for e in rows} for sp,rows in splits.items()}
    leakage=len((split_sets['train'] & split_sets['dev']) | (split_sets['train'] & split_sets['test']) | (split_sets['dev'] & split_sets['test']))
    if leakage: raise ValueError('RxCUI split leakage detected')
    report={'records':len(records),'selected_codes':len(selected),'max_codes':ns.max_codes,'examples_per_code':ns.examples_per_code,'seed':ns.seed,'splits':{k:len(v) for k,v in splits.items()},'codes_by_split':dict(Counter(code_split.values())),'strata':dict(Counter(stratum(r) for r in selected)),'split_leakage':leakage,'official_evaluation':False,'synthetic_pilot':True,'exact_alias_diagnostic_excluded_from_main_metric':True}
    if ns.dry_run:
        print(json.dumps(report, indent=2)); return
    out=Path(ns.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps({'splits':splits,'report':report}, ensure_ascii=False, indent=2), encoding='utf-8'); print(json.dumps({'output':str(out), **report}, indent=2))
if __name__=='__main__': main()
