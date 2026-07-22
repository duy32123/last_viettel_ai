from __future__ import annotations
from pathlib import Path
import json
from src.data.kb_schema import read_jsonl
from .retrieval import LexicalIndex, report_records

def load_jsonl(path): return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]
def write_jsonl(path, rows): Path(path).write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n', encoding='utf-8')
def validate_kb(records, production=False):
    if not records: raise ValueError('KB empty')
    terms={(r.terminology,r.version) for r in records}
    if production and any(not r.verified for r in records): raise ValueError('production KB contains verified=false')
    for r in records: r.validate()
    if len(records)!=len({(r.terminology,r.version,r.code) for r in records}): raise ValueError('duplicate KB codes')
    return report_records(records)
def link_rows(rows, index:LexicalIndex, top_k=5):
    out=[]
    for r in rows:
        text=r['text']; ents=[]
        for e in r.get('entities',[]):
            ent=dict(e); s,en=ent.get('position',[ent.get('start'),ent.get('end')])
            if text[s:en] != ent['text']: raise ValueError('entity offset invariant failed')
            cands=[c.to_dict() for c in index.search(ent['text'], ent['type'], top_k)]
            ent['candidates']=cands; ent['candidate_missing']=not bool(cands); ents.append(ent)
        out.append({**r,'entities':ents})
    return out
