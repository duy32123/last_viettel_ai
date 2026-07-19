from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.kb_schema import read_jsonl

def split_code(code):
    h=int(hashlib.sha256(code.encode()).hexdigest(),16)%100
    return 'train' if h<80 else ('dev' if h<90 else 'test')

def context_for(rec):
    tty=rec.metadata.get('TTY'); name=rec.canonical_name
    return [
        {'task':'exact_alias_diagnostic','mention':name,'context':name,'positive_code':rec.code,'language':'en','tty':tty},
        {'task':'noisy_medication_mention_to_rxcui','mention':f'thuốc {name}','context':f'đang sử dụng thuốc {name} mỗi ngày','positive_code':rec.code,'language':'vi','tty':tty},
        {'task':'mention_plus_dose_context_to_rxcui','mention':name,'context':f'{name} 500 mg đường uống','positive_code':rec.code,'language':'vi','tty':tty},
        {'task':'brand_vs_ingredient_disambiguation','mention':name,'context':f'đang sử dụng {name} mỗi ngày','positive_code':rec.code,'language':'vi','tty':tty},
    ]

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--kb-dir', default='data/processed/linking_kb_rxnorm'); p.add_argument('--output', default='data/processed/linking_kb_rxnorm/rxnorm_pilot_examples.json'); p.add_argument('--dry-run', action='store_true'); ns=p.parse_args(argv)
    records=[]
    for path in Path(ns.kb_dir).glob('*.jsonl'): records.extend(read_jsonl(path))
    splits={'train':[],'dev':[],'test':[]}; assigned={}
    for rec in records:
        sp=split_code(rec.code); assigned[rec.code]=sp
        for i,ex in enumerate(context_for(rec)):
            splits[sp].append({'id':f'{rec.code}:{i}', **ex, 'official_evaluation':False})
    if ns.dry_run:
        print(json.dumps({'records':len(records),'splits':{k:len(v) for k,v in splits.items()},'official_evaluation':False}, indent=2)); return
    out=Path(ns.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding='utf-8'); print(json.dumps({'output':str(out),'splits':{k:len(v) for k,v in splits.items()},'official_evaluation':False}, indent=2))
if __name__=='__main__': main()
