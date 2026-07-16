from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .labels import TARGET_TYPES, IGNORE_TYPES, LABEL2ID, ID2LABEL, entity_to_bilou, labels_to_spans

@dataclass
class PreprocessStats:
    records:int=0
    chunks:int=0
    dropped_boundary_entities:int=0
    ignored_entities:int=0
    overlapping_entities:int=0
    duplicate_entities:int=0
    partial_entity_chunks:int=0


def read_jsonl(path: str|Path) -> list[dict[str,Any]]:
    p=Path(path)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def validate_entities(record: dict[str,Any], stats: PreprocessStats) -> list[dict[str,Any]]:
    entities=[]; seen=set(); occupied=[]
    for i,e in enumerate(sorted(record.get("entities", []), key=lambda x:(x["start"], x["end"]))):
        if e["type"] in IGNORE_TYPES:
            stats.ignored_entities += 1; continue
        if e["type"] not in TARGET_TYPES: continue
        key=(e["start"], e["end"], e["type"])
        if key in seen:
            stats.duplicate_entities += 1; continue
        seen.add(key)
        for s,en,_ in occupied:
            if max(s,e["start"]) < min(en,e["end"]):
                stats.overlapping_entities += 1
                raise ValueError(f"overlap/nested entity unsupported in {record.get('id')}: {e}")
        ent=dict(e); ent["_key"]=key; ent["_ordinal"]=i
        occupied.append((e["start"], e["end"], e["type"])); entities.append(ent)
    return entities


def _call_tokenizer(tokenizer, text, max_length, stride):
    return tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length, stride=stride, return_overflowing_tokens=True, padding=False)


def _as_chunks(enc) -> list[dict[str,Any]]:
    if hasattr(enc, "encodings") and enc.encodings:
        chunks=[]
        for e in enc.encodings:
            chunks.append({"input_ids": e.ids, "attention_mask": getattr(e, "attention_mask", [1]*len(e.ids)), "offset_mapping": e.offsets, "overflow_to_sample_mapping": 0})
        return chunks
    offs=enc["offset_mapping"]
    if offs and isinstance(offs[0], tuple):
        return [{k:v for k,v in enc.items()}]
    chunks=[]; n=len(offs)
    for i in range(n):
        chunks.append({k:(v[i] if isinstance(v, list) and len(v)==n else v) for k,v in enc.items()})
    return chunks


def label_chunk(text: str, entities: list[dict[str,Any]], offsets: list[tuple[int,int]], stats: PreprocessStats) -> tuple[list[int], set[tuple[int,int,str]]]:
    labels=[-100 if s==e else LABEL2ID["O"] for s,e in offsets]
    fully_covered=set()
    for ent in entities:
        overlapping=[i for i,(s,e) in enumerate(offsets) if s!=e and max(s, ent["start"]) < min(e, ent["end"])]
        if not overlapping:
            continue
        token_idxs=[i for i in overlapping if offsets[i][0] >= ent["start"] and offsets[i][1] <= ent["end"]]
        full = bool(token_idxs) and offsets[token_idxs[0]][0] == ent["start"] and offsets[token_idxs[-1]][1] == ent["end"]
        if not full:
            stats.partial_entity_chunks += 1
            for idx in overlapping:
                labels[idx] = -100
            continue
        fully_covered.add(ent["_key"])
        for idx, lab in zip(token_idxs, entity_to_bilou(len(token_idxs), ent["type"])):
            labels[idx]=LABEL2ID[lab]
    return labels, fully_covered


def preprocess_records(records: list[dict[str,Any]], tokenizer, max_length:int, stride:int) -> tuple[list[dict[str,Any]], PreprocessStats]:
    stats=PreprocessStats(records=len(records)); features=[]
    for rec in records:
        entities=validate_entities(rec, stats); covered=set()
        enc=_call_tokenizer(tokenizer, rec["text"], max_length, stride)
        for chunk_idx, ch in enumerate(_as_chunks(enc)):
            offsets=[tuple(x) for x in ch["offset_mapping"]]
            labels, chunk_covered=label_chunk(rec["text"], entities, offsets, stats)
            covered |= chunk_covered
            features.append({"input_ids":ch["input_ids"], "attention_mask":ch.get("attention_mask", [1]*len(ch["input_ids"])), "labels":labels, "offset_mapping":offsets, "record_id":rec.get("id"), "text":rec["text"], "chunk_index":chunk_idx, "gold_entities":[{k:v for k,v in e.items() if not k.startswith("_")} for e in entities]})
            stats.chunks += 1
        missing={e["_key"] for e in entities} - covered
        stats.dropped_boundary_entities += len(missing)
    return features, stats


def decode_feature_spans(feature: dict[str,Any], label_ids: list[int], id2label: dict[int,str] | dict[str,str] | None=None) -> list[dict[str,Any]]:
    mapping=id2label or ID2LABEL
    labs=[]
    for i in label_ids:
        if i == -100: labs.append(-100)
        else: labs.append(mapping.get(i, mapping.get(str(i))))
    spans=labels_to_spans(feature["offset_mapping"], labs)
    for s in spans: s["text"]=feature["text"][s["start"]:s["end"]]
    return spans
