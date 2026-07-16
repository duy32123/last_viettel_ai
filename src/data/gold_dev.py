from __future__ import annotations
import json
from pathlib import Path
from src.data.dataset_schema import validate_record, VALID_TYPES
from src.models.ner.preprocess import PreprocessStats, validate_entities


def promote(todo_path: Path, out_path: Path) -> dict:
    approved=[]; skipped=0
    for line in todo_path.read_text(encoding='utf-8').splitlines() if todo_path.exists() else []:
        if not line.strip(): continue
        r=json.loads(line)
        if r.get('review_status') != 'approved': skipped += 1; continue
        rec={k:v for k,v in r.items() if k not in {'review_status','reviewer','review_notes','proposed_entities'}}
        rec['entities']=r.get('proposed_entities', [])
        rec.setdefault('metadata', {})['gold_evaluation']=True
        validate_record(rec)
        validate_entities(rec, PreprocessStats())
        if any(e['type'] not in VALID_TYPES for e in rec.get('entities', [])): raise ValueError('gold dev contains non-target entity type')
        approved.append(rec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as fh:
        for r in approved: fh.write(json.dumps(r, ensure_ascii=False)+'\n')
    return {'approved':len(approved),'skipped':skipped,'output':str(out_path)}
