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

def labels_to_spans(offsets: list[tuple[int,int]], labels: list[str]) -> list[dict]:
    spans=[]; i=0
    while i < len(labels):
        lab=labels[i]
        if lab in {"O", -100} or lab == -100:
            i+=1; continue
        if isinstance(lab, int): raise ValueError("labels_to_spans expects label strings")
        prefix, typ = lab.split("-",1)
        if prefix == "U":
            s,e=offsets[i]; spans.append({"start":s,"end":e,"type":typ}); i+=1; continue
        if prefix != "B":
            i+=1; continue
        start=offsets[i][0]; j=i+1; end=offsets[i][1]
        while j < len(labels):
            nxt=labels[j]
            if not isinstance(nxt, str) or "-" not in nxt: break
            p,t=nxt.split("-",1)
            if t != typ: break
            end=offsets[j][1]
            if p == "L":
                j+=1; break
            if p != "I": break
            j+=1
        spans.append({"start":start,"end":end,"type":typ}); i=max(j, i+1)
    return spans
