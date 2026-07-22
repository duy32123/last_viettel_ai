from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from src.models.llm_hybrid.extractor import ASSERTION_TYPES, ALLOWED_ASSERTIONS, SYSTEM_PROMPT
from src.models.ner.labels import TARGET_TYPES


def _sort_key(path: Path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


def _target(text: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target=[]
    for index,row in enumerate(rows):
        if row.get('type') not in TARGET_TYPES:
            raise ValueError(f'row {index}: unsupported type {row.get("type")!r}')
        position=row.get('position')
        if not isinstance(position,list) or len(position)!=2:
            raise ValueError(f'row {index}: position must be [start,end]')
        start,end=position
        if not (isinstance(start,int) and isinstance(end,int) and 0 <= start < end <= len(text)):
            raise ValueError(f'row {index}: invalid position {position!r}')
        if text[start:end] != row.get('text'):
            raise ValueError(f'row {index}: exact offset invariant failed')
        item={'text':row['text'],'type':row['type'],'start':start,'end':end}
        if row['type'] in ASSERTION_TYPES:
            supplied=row.get('assertions',[])
            if not isinstance(supplied,list): supplied=[]
            item['assertions']=[name for name in ALLOWED_ASSERTIONS if name in supplied]
        target.append(item)
    return sorted(target, key=lambda item:(item['start'],item['end'],item['type']))


def build_record(text: str, rows: list[dict[str, Any]], record_id: str) -> dict[str, Any]:
    target=_target(text,rows)
    return {
        'id':record_id,
        'messages':[
            {'role':'system','content':SYSTEM_PROMPT},
            {'role':'user','content':f'VĂN BẢN (độ dài {len(text)} ký tự):\n{text}'},
            {'role':'assistant','content':json.dumps(target,ensure_ascii=False,separators=(',',':'))},
        ],
        'source_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'entity_count':len(target),
    }


def build_dataset(input_dir: Path, annotations_dir: Path) -> list[dict[str, Any]]:
    records=[]
    for input_path in sorted(input_dir.glob('*.txt'),key=_sort_key):
        annotation_path=annotations_dir/(input_path.stem+'.json')
        if not annotation_path.exists():
            raise FileNotFoundError(f'missing annotation for {input_path.name}: {annotation_path}')
        text=input_path.read_text(encoding='utf-8')
        rows=json.loads(annotation_path.read_text(encoding='utf-8'))
        if not isinstance(rows,list): raise ValueError(f'{annotation_path}: expected JSON array')
        records.append(build_record(text,rows,input_path.stem))
    if not records: raise FileNotFoundError(f'no *.txt files in {input_dir}')
    return records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description='Build leakage-safe chat SFT data from BTC input/ground truth pairs.')
    parser.add_argument('--input-dir',required=True)
    parser.add_argument('--annotations-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--source-role',required=True,choices=['released_gold','manual_gold'],help='Safety declaration: official evaluation inputs are forbidden.')
    parser.add_argument('--dev-ratio',type=float,default=0.2)
    parser.add_argument('--seed',type=int,default=20260722)
    ns=parser.parse_args(argv)
    if not 0 < ns.dev_ratio < 1: raise ValueError('--dev-ratio must be between 0 and 1')
    records=build_dataset(Path(ns.input_dir),Path(ns.annotations_dir))
    shuffled=list(records); random.Random(ns.seed).shuffle(shuffled)
    dev_count=max(1,round(len(shuffled)*ns.dev_ratio)); dev_ids={row['id'] for row in shuffled[:dev_count]}
    train=[row for row in records if row['id'] not in dev_ids]
    dev=[row for row in records if row['id'] in dev_ids]
    out=Path(ns.output_dir); _write_jsonl(out/'train.jsonl',train); _write_jsonl(out/'dev.jsonl',dev)
    manifest={'documents':len(records),'train_documents':len(train),'dev_documents':len(dev),'seed':ns.seed,'dev_ratio':ns.dev_ratio,'source_role':ns.source_role,'official_test_used':False,'train_ids':[r['id'] for r in train],'dev_ids':[r['id'] for r in dev],'candidates_in_target':False}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__': main()