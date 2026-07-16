from __future__ import annotations
from typing import Any
from .labels import to_vector

def insert_entity_markers(text: str, entity: dict[str,Any], max_chars: int=800) -> str:
    start,end=entity["position"] if "position" in entity else (entity["start"], entity["end"])
    if text[start:end] != entity["text"]: raise ValueError("entity offset invariant failed")
    left=max(0, start-max_chars//2); right=min(len(text), end+max_chars//2)
    prefix=text[left:start]; suffix=text[end:right]
    return f"<TYPE_{entity['type']}> {prefix}<ENT_START> {entity['text']} <ENT_END>{suffix}"

def make_examples(records: list[dict[str,Any]], max_chars: int=800) -> list[dict[str,Any]]:
    examples=[]
    for rec in records:
        text=rec["text"]
        for i,e in enumerate(rec.get("entities", [])):
            pos=e.get("position", [e.get("start"), e.get("end")])
            ent={**e,"position":pos}
            examples.append({"id":f"{rec.get('id','doc')}::e{i}","record_id":rec.get("id"),"input_text":insert_entity_markers(text, ent, max_chars),"labels":to_vector(e.get("assertions", [])),"entity":ent,"text":text})
    return examples

def tokenize_examples(examples, tokenizer, max_length:int=256):
    enc=tokenizer([e["input_text"] for e in examples], truncation=True, padding=True, max_length=max_length)
    enc["labels"]=[e["labels"] for e in examples]
    return enc
