from __future__ import annotations
import re
from typing import Any
from .preprocess import preprocess_records, decode_feature_spans, normalize_predicted_span

LAB_RESULT_TYPE="KẾT_QUẢ_XÉT_NGHIỆM"
LAB_TEST_TYPE="TÊN_XÉT_NGHIỆM"
_LAB_TRIGGERS=("xét nghiệm", "xn", "kết quả", "chỉ số", "máu", "huyết thanh", "nước tiểu")
_CLAUSE_BOUNDARY_RE=re.compile(r"[\n\r.;!?]")
_NON_LAB_PATTERNS=[
    re.compile(r"^\d{1,2}m\d{2}$", re.I),
    re.compile(r"^\d{2,3}\s*cm$", re.I),
    re.compile(r"^\d{1,3}(?:[,.]\d+)?\s*kg$", re.I),
    re.compile(r"^\d{1,3}\s*tuổi$", re.I),
    re.compile(r"^\d+(?:[,.]\d+)?\s*(?:mg|ml|mL)$", re.I),
    re.compile(r"^\d{2,3}/\d{2,3}\s*mmhg$", re.I),
    re.compile(r"^\d{2,3}\s*(?:bpm|lần/phút)$", re.I),
    re.compile(r"^\d{2}(?:[,.]\d+)?\s*(?:c|°c)$", re.I),
    re.compile(r"^\d{1,2}:\d{2}$"),
    re.compile(r"^\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?$"),
]
_PERCENT_RE=re.compile(r"^\d+(?:[,.]\d+)?%$")
_QUAL_RE=re.compile(r"^(?:cao|thấp|tăng|giảm|dương tính|âm tính)$", re.I)
_PRICE_CONTEXT_RE=re.compile(r"giá|giảm giá|khuyến mãi|chiết khấu|rẻ|đắt", re.I)


def _clause_bounds(text: str, start: int, end: int) -> tuple[int,int]:
    left=0
    for m in _CLAUSE_BOUNDARY_RE.finditer(text, 0, start):
        left=m.end()
    right=len(text)
    m=_CLAUSE_BOUNDARY_RE.search(text, end)
    if m: right=m.start()
    return left, right


def _has_lab_trigger(text: str, start: int, end: int, window: int=80) -> bool:
    left=max(0, start-window); right=min(len(text), end+window)
    local=text[left:right].casefold()
    return any(t in local for t in _LAB_TRIGGERS)


def _has_predicted_test_evidence(text: str, span: dict[str,Any], spans: list[dict[str,Any]], window: int=80) -> bool:
    start=int(span["start"]); end=int(span["end"])
    clause_start, clause_end=_clause_bounds(text, start, end)
    for other in spans:
        if other is span or other.get("type") != LAB_TEST_TYPE:
            continue
        os=int(other["start"]); oe=int(other["end"])
        same_clause = clause_start <= os < oe <= clause_end
        nearby = abs(os - end) <= window or abs(start - oe) <= window
        if same_clause or nearby:
            return True
    return False


def _looks_like_non_lab_measurement(text: str, span: dict[str,Any]) -> bool:
    surface=str(span.get("text") or text[span["start"]:span["end"]]).strip()
    folded=surface.casefold()
    if any(p.match(surface) for p in _NON_LAB_PATTERNS):
        return True
    if _PERCENT_RE.match(surface):
        left=max(0, span["start"]-50); right=min(len(text), span["end"]+50)
        return bool(_PRICE_CONTEXT_RE.search(text[left:right]))
    if _QUAL_RE.match(folded):
        return True
    return False


def filter_non_lab_result_predictions(text: str, spans: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """Drop non-lab measurement false positives after final span merge.

    This function never changes valid span boundaries and never creates spans. It only
    removes KẾT_QUẢ_XÉT_NGHIỆM predictions that look like ordinary measurements and
    lack lab evidence from a predicted test-name span or local lab trigger context.
    """
    kept=[]
    for span in spans:
        if not (0 <= int(span.get("start", -1)) < int(span.get("end", -1)) <= len(text)):
            continue
        if text[span["start"]:span["end"]] != span.get("text", text[span["start"]:span["end"]]):
            span=dict(span); span["text"]=text[span["start"]:span["end"]]
        if not span["text"]:
            continue
        if span.get("type") == LAB_RESULT_TYPE:
            has_evidence=_has_predicted_test_evidence(text, span, spans) or _has_lab_trigger(text, span["start"], span["end"])
            if not has_evidence and _looks_like_non_lab_measurement(text, span):
                continue
        kept.append(span)
    return kept


def finalize_predictions(text: str, chunks: list[list[dict[str,Any]]] | list[dict[str,Any]]) -> list[dict[str,Any]]:
    if chunks and isinstance(chunks[0], dict):
        merged=merge_chunk_predictions([chunks], text)  # type: ignore[list-item]
    else:
        merged=merge_chunk_predictions(chunks, text)  # type: ignore[arg-type]
    return filter_non_lab_result_predictions(text, merged)


def merge_chunk_predictions(chunks: list[list[dict[str,Any]]], text: str | None=None) -> list[dict[str,Any]]:
    best_by_boundary={}
    for spans in chunks:
        for span in spans:
            s=dict(span)
            if text is not None:
                norm=normalize_predicted_span(text, s["start"], s["end"])
                if not norm: continue
                s.update(norm)
            boundary=(s["start"], s["end"])
            cur=best_by_boundary.get(boundary)
            if cur is None or s.get("score", 0.0) > cur.get("score", 0.0):
                best_by_boundary[boundary]=s
    return [best_by_boundary[k] for k in sorted(best_by_boundary)]


def _model_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"


def _span_scores(offsets, pred_ids, token_scores, spans):
    for span in spans:
        idxs=[i for i,(s,e) in enumerate(offsets) if s!=e and s >= span["start"] and e <= span["end"]]
        if idxs:
            span["score"] = min(float(token_scores[i][pred_ids[i]]) for i in idxs)
        else:
            span["score"] = 0.0
    return spans


def predict_with_model(text: str, tokenizer, model, id2label: dict[int,str] | dict[str,str] | None=None, max_length:int=256, stride:int=64) -> list[dict[str,Any]]:
    try:
        import torch
    except Exception as e:
        raise RuntimeError("torch is required for model inference") from e
    model.eval()
    device=_model_device(model)
    config_id2label=id2label or getattr(model.config, "id2label", None)
    if not config_id2label: raise ValueError("model.config.id2label is required for NER inference")
    rec={"id":"doc","text":text,"entities":[]}
    features,_=preprocess_records([rec], tokenizer, max_length, stride)
    chunk_spans=[]
    for f in features:
        inputs={"input_ids":torch.tensor([f["input_ids"]], device=device), "attention_mask":torch.tensor([f["attention_mask"]], device=device)}
        with torch.no_grad():
            logits=model(**inputs).logits[0]
            probs=torch.softmax(logits, dim=-1)
        ids=logits.argmax(dim=-1).detach().cpu().tolist()
        scores=probs.detach().cpu().tolist()
        spans=decode_feature_spans(f, ids, config_id2label)
        spans=_span_scores(f["offset_mapping"], ids, scores, spans)
        for s in spans:
            s["text"]=text[s["start"]:s["end"]]
            if text[s["start"]:s["end"]] != s["text"]: raise ValueError("prediction offset invariant failed")
        chunk_spans.append(spans)
    return finalize_predictions(text, chunk_spans)
