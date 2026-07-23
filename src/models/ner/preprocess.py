from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from .labels import LABEL2ID, ID2LABEL, labels_to_spans, TARGET_TYPES

@dataclass
class PreprocessStats:
    ignored_entities:int=0; duplicate_entities:int=0; dropped_boundary_entities:int=0; partial_entity_chunks:int=0

def validate_entities(rec:dict[str,Any], stats:PreprocessStats|None=None):
    stats=stats or PreprocessStats(); seen=set(); occupied=[]; out=[]; text=rec.get('text','')
    for e in rec.get('entities',[]):
        typ=e.get('type')
        if typ in {'IGNORE','UNMAPPED'} or typ not in TARGET_TYPES:
            stats.ignored_entities += 1; continue
        s=e.get('start', e.get('position',[None,None])[0]); en=e.get('end', e.get('position',[None,None])[1])
        if not isinstance(s,int) or not isinstance(en,int) or not (0 <= s < en <= len(text)) or text[s:en] != e.get('text', text[s:en]):
            raise ValueError('invalid entity span')
        key=(s,en,typ)
        if key in seen: stats.duplicate_entities += 1; continue
        if any(not (en <= a or s >= b) for a,b in occupied): raise ValueError('overlapping entities')
        seen.add(key); occupied.append((s,en)); out.append({'start':s,'end':en,'type':typ,'text':text[s:en]})
    return out

def _label_sequence(offsets, entities, stats):
    labels=[LABEL2ID['O'] if s!=e else -100 for s,e in offsets]
    for ent in entities:
        toks=[i for i,(s,e) in enumerate(offsets) if s!=e and max(s,ent['start']) < min(e,ent['end'])]
        if not toks: continue
        if offsets[toks[0]][0] != ent['start'] or offsets[toks[-1]][1] != ent['end']:
            for i in toks: labels[i] = -100
            stats.partial_entity_chunks += 1
            continue
        if len(toks)==1: labels[toks[0]]=LABEL2ID[f'U-{ent["type"]}']
        else:
            labels[toks[0]]=LABEL2ID[f'B-{ent["type"]}']; labels[toks[-1]]=LABEL2ID[f'L-{ent["type"]}']
            for i in toks[1:-1]: labels[i]=LABEL2ID[f'I-{ent["type"]}']
    return labels

def preprocess_records(records, tokenizer, max_length=256, stride=64):
    features=[]; stats=PreprocessStats()
    for rec in records:
        ents=validate_entities(rec, stats); text=rec['text']
        enc=tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length, stride=stride, return_overflowing_tokens=True, padding=False)
        for idx,offsets in enumerate(enc['offset_mapping']):
            f={k:enc[k][idx] for k in ('input_ids','attention_mask','offset_mapping') if k in enc}
            f.update({'record_id':rec.get('id'),'text':text,'gold_entities':ents})
            f['labels']=_label_sequence(offsets, ents, stats)
            features.append(f)
    return features, stats

def decode_feature_spans(feature, label_ids):
    labs=[ID2LABEL.get(int(i),'O') if int(i) != -100 else 'O' for i in label_ids]
    spans=labels_to_spans(feature['offset_mapping'], labs, len(feature['text']))
    out=[]
    for s in spans:
        txt=feature['text'][s['start']:s['end']]
        if txt: out.append({**s,'text':txt})
    return out


def read_jsonl(path):
    import json
    from pathlib import Path
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]
