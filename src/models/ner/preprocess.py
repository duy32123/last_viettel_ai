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


def read_jsonl(path: str|Path) -> list[dict[str,Any]]:
    p=Path(path)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def validate_entities(record: dict[str,Any], stats: PreprocessStats) -> list[dict[str,Any]]:
    entities=[]; seen=set(); occupied=[]
    for e in sorted(record.get("entities", []), key=lambda x:(x["start"], x["end"])):
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
        occupied.append((e["start"], e["end"], e["type"])); entities.append(e)
    return entities


def _call_tokenizer(tokenizer, text, max_length, stride):
    return tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length, stride=stride, return_overflowing_tokens=True, padding=False)


def _as_chunks(enc) -> list[dict[str,Any]]:
    if hasattr(enc, "encodings") and enc.encodings:
        chunks=[]
        for i, e in enumerate(enc.encodings):
            d={"input_ids": e.ids, "attention_mask": getattr(e, "attention_mask", [1]*len(e.ids)), "offset_mapping": e.offsets, "overflow_to_sample_mapping": 0}
            chunks.append(d)
        return chunks
    offs=enc["offset_mapping"]
    if offs and isinstance(offs[0], tuple):
        return [{k:v for k,v in enc.items()}]
    chunks=[]
    n=len(offs)
    for i in range(n):
        chunks.append({k:(v[i] if isinstance(v, list) and len(v)==n else v) for k,v in enc.items()})
    return chunks


def label_chunk(text: str, entities: list[dict[str,Any]], offsets: list[tuple[int,int]], stats: PreprocessStats) -> list[int]:
    labels=[-100 if s==e else LABEL2ID["O"] for s,e in offsets]
    for ent in entities:
        token_idxs=[i for i,(s,e) in enumerate(offsets) if s!=e and s >= ent["start"] and e <= ent["end"]]
        if not token_idxs:
            if any(s!=e and max(s,ent["start"]) < min(e,ent["end"]) for s,e in offsets): stats.dropped_boundary_entities += 1
            continue
        if offsets[token_idxs[0]][0] != ent["start"] or offsets[token_idxs[-1]][1] != ent["end"]:
            stats.dropped_boundary_entities += 1; continue
        for idx, lab in zip(token_idxs, entity_to_bilou(len(token_idxs), ent["type"])):
            labels[idx]=LABEL2ID[lab]
    return labels


def preprocess_records(records: list[dict[str,Any]], tokenizer, max_length:int, stride:int) -> tuple[list[dict[str,Any]], PreprocessStats]:
    stats=PreprocessStats(records=len(records)); features=[]
    for rec in records:
        entities=validate_entities(rec, stats)
        enc=_call_tokenizer(tokenizer, rec["text"], max_length, stride)
        for chunk_idx, ch in enumerate(_as_chunks(enc)):
            offsets=[tuple(x) for x in ch["offset_mapping"]]
            labels=label_chunk(rec["text"], entities, offsets, stats)
            features.append({"input_ids":ch["input_ids"], "attention_mask":ch.get("attention_mask", [1]*len(ch["input_ids"])), "labels":labels, "offset_mapping":offsets, "record_id":rec.get("id"), "text":rec["text"], "chunk_index":chunk_idx})
            stats.chunks += 1
    return features, stats


def decode_feature_spans(feature: dict[str,Any], label_ids: list[int]) -> list[dict[str,Any]]:
    labs=[ID2LABEL[i] if i != -100 else -100 for i in label_ids]
    spans=labels_to_spans(feature["offset_mapping"], labs)
    for s in spans: s["text"]=feature["text"][s["start"]:s["end"]]
    return spans
