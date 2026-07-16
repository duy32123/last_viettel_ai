from __future__ import annotations
from pathlib import Path
from src.data.dataset_schema import validate_record


def _flush(tokens, labels, rows, source, split, license_text, mapping):
    if not tokens: return
    text=" ".join(tokens); offsets=[]; pos=0
    for tok in tokens:
        offsets.append((pos,pos+len(tok))); pos += len(tok)+1
    ents=[]; cur=None
    for i, lab in enumerate(labels):
        if lab == "O": mapped="IGNORE"; prefix="O"; raw="O"
        else:
            prefix, raw = lab.split("-",1) if "-" in lab else ("B", lab)
            mapped=mapping.get(raw, "UNMAPPED")
        if mapped == "IGNORE":
            if cur: ents.append(cur); cur=None
            continue
        if prefix == "B" or cur is None or cur["type"] != mapped:
            if cur: ents.append(cur)
            s,e=offsets[i]; cur={"id":"","start":s,"end":e,"text":text[s:e],"type":mapped,"assertions":[],"candidates":[],"metadata":{"source_label":raw}}
        else:
            cur["end"]=offsets[i][1]; cur["text"]=text[cur["start"]:cur["end"]]
    if cur: ents.append(cur)
    for j,e in enumerate(ents,1): e["id"]=f"E{j}"
    rec={"id":f"phoner_{len(rows)+1}","text":text,"entities":ents,"relations":[],"metadata":{},"source":source,"source_split":split,"license":license_text}
    validate_record(rec); rows.append(rec)


def load_native(path: Path, source: str, split: str, license_text: str, mapping: dict[str,str]) -> list[dict]:
    rows=[]; toks=[]; labs=[]
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line=line.strip()
            if not line:
                _flush(toks,labs,rows,source,split,license_text,mapping); toks=[]; labs=[]; continue
            parts=line.split()
            toks.append(parts[0]); labs.append(parts[-1])
    _flush(toks,labs,rows,source,split,license_text,mapping)
    return rows
