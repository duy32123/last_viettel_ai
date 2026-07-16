from __future__ import annotations
TARGET_TYPES=["TRIỆU_CHỨNG","TÊN_XÉT_NGHIỆM","KẾT_QUẢ_XÉT_NGHIỆM","CHẨN_ĐOÁN","THUỐC"]
IGNORE_TYPES={"IGNORE","UNMAPPED"}
OUTSIDE="O"

def bilou_labels(types: list[str] | None=None) -> list[str]:
    types=types or TARGET_TYPES
    labels=[OUTSIDE]
    for t in types:
        labels.extend([f"B-{t}", f"I-{t}", f"L-{t}", f"U-{t}"])
    return labels

LABELS=bilou_labels()
LABEL2ID={l:i for i,l in enumerate(LABELS)}
ID2LABEL={i:l for l,i in LABEL2ID.items()}

def entity_to_bilou(length: int, typ: str) -> list[str]:
    if length <= 0: raise ValueError("entity token length must be positive")
    if length == 1: return [f"U-{typ}"]
    return [f"B-{typ}"] + [f"I-{typ}"]*(length-2) + [f"L-{typ}"]

def _valid_offset(offset: tuple[int,int]) -> bool:
    s,e=offset
    return isinstance(s,int) and isinstance(e,int) and 0 <= s < e

def labels_to_spans(offsets: list[tuple[int,int]], labels: list[str], text_length: int | None=None) -> list[dict]:
    spans=[]; i=0
    n=len(labels)
    while i < n:
        lab=labels[i]
        if lab in {"O", -100} or lab == -100 or lab is None:
            i+=1; continue
        if isinstance(lab, int): raise ValueError("labels_to_spans expects label strings")
        if "-" not in lab:
            i+=1; continue
        prefix, typ = lab.split("-",1)
        if typ not in TARGET_TYPES:
            i+=1; continue
        if prefix == "U":
            if _valid_offset(offsets[i]):
                s,e=offsets[i]
                if text_length is None or e <= text_length: spans.append({"start":s,"end":e,"type":typ})
            i+=1; continue
        if prefix != "B" or not _valid_offset(offsets[i]):
            i+=1; continue
        start=offsets[i][0]; j=i+1; last_end=offsets[i][1]; valid=False
        # Valid multi-token entity is B (I*) L; incomplete B or B-I* without L is dropped.
        while j < n:
            nxt=labels[j]
            if not isinstance(nxt, str) or "-" not in nxt or not _valid_offset(offsets[j]): break
            p,t=nxt.split("-",1)
            if t != typ: break
            if p == "I":
                last_end=offsets[j][1]; j+=1; continue
            if p == "L":
                last_end=offsets[j][1]; valid=True; j+=1; break
            break
        if valid and start < last_end and (text_length is None or last_end <= text_length):
            spans.append({"start":start,"end":last_end,"type":typ})
        i=max(j, i+1)
    return spans
