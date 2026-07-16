from __future__ import annotations
import json, re, hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from src.data.dataset_schema import validate_record, VALID_TYPES
from src.data.synthetic.generator import ann_hash

DRUG_ALLOW={"prozac","zoloft","amlodipine","panadol","paracetamol","ibuprofen","metformin","insulin"}
DRUG_IGNORE={"inr","vi khuẩn","ruồi giấm","drosophila","bacteria"}
DIAG_ALLOW={"béo phì","tăng huyết áp","đái tháo đường","obesity","hypertension","diabetes"}
SYMPTOM_ALLOW={"sốt","ho","ho khan","khó thở","đau ngực","mệt","đau bụng"}
IGNORE_TERMS={"vi khuẩn","ruồi giấm","drosophila","bacteria"}
TEST_NAMES={"glucose","wbc","inr","hba1c","hemoglobin","creatinine","crp","ast","alt"}
RESULT_WORDS={"cao","thấp","dương tính","âm tính","tăng","giảm","positive","negative","high","low"}
TARGET_TYPES=set(VALID_TYPES)

@dataclass
class WeakDecision:
    target: str
    confidence: float
    method: str
    reason: str


def context_window(text: str, start: int, end: int, radius: int=80) -> str:
    return text[max(0,start-radius):min(len(text), end+radius)]


def _norm(s: str) -> str:
    return " ".join(s.casefold().strip().split())


def classify_candidate(text: str, source_label: str, start: int, end: int, qwen_enabled: bool=False) -> WeakDecision:
    if not (0 <= start < end <= len(text)):
        return WeakDecision("REJECT",0.0,"rule_reject","invalid_offset")
    mention=text[start:end]
    if not mention.strip(): return WeakDecision("REJECT",0.0,"rule_reject","empty_mention")
    m=_norm(mention); ctx=_norm(context_window(text,start,end))
    label=source_label.upper()
    if label == "DRUGCHEMICAL":
        if m in DRUG_ALLOW: return WeakDecision("THUỐC",0.98,"rule_allowlist","known_drug")
        if m in DRUG_IGNORE: return WeakDecision("IGNORE",0.99,"rule_ignore","not_drug")
        return WeakDecision("REVIEW",0.50,"qwen_pending" if qwen_enabled else "rule_ambiguous","ambiguous_drugchemical")
    if label == "DISEASESYMTOM":
        if m in IGNORE_TERMS: return WeakDecision("IGNORE",0.99,"rule_ignore","organism_or_non_clinical")
        if m in DIAG_ALLOW: return WeakDecision("CHẨN_ĐOÁN",0.96,"rule_allowlist","known_diagnosis")
        if m in SYMPTOM_ALLOW: return WeakDecision("TRIỆU_CHỨNG",0.95,"rule_allowlist","known_symptom")
        return WeakDecision("REVIEW",0.50,"qwen_pending" if qwen_enabled else "rule_ambiguous","ambiguous_disease_symptom")
    if label == "DIAGNOSTICS":
        if m in TEST_NAMES: return WeakDecision("TÊN_XÉT_NGHIỆM",0.96,"rule_allowlist","known_test_name")
        return WeakDecision("IGNORE",0.90,"rule_ignore","diagnostic_not_target")
    if label == "UNITCALIBRATOR":
        has_test=any(t in ctx for t in TEST_NAMES)
        looks_result=bool(re.search(r"\d", mention)) or m in RESULT_WORDS
        if has_test and looks_result: return WeakDecision("KẾT_QUẢ_XÉT_NGHIỆM",0.93,"rule_context","result_attached_to_test")
        return WeakDecision("IGNORE",0.91,"rule_ignore","unit_or_qualifier_without_test")
    return WeakDecision("IGNORE",0.99,"rule_ignore","unsupported_source_label")


def _source_entities(rec: dict[str,Any]) -> list[dict[str,Any]]:
    if rec.get('source_entities'): return rec['source_entities']
    out=[]
    for e in rec.get('entities', []):
        label=e.get('source_label') or e.get('metadata',{}).get('source_label') or e.get('type')
        out.append({'start':e['start'],'end':e['end'],'text':e.get('text', rec['text'][e['start']:e['end']]),'source_label':label})
    return out


def _overlaps(a,b): return max(a['start'],b['start']) < min(a['end'],b['end'])


def weak_label_records(rows: list[dict[str,Any]], min_confidence: float=0.9, qwen_enabled: bool=False) -> tuple[list[dict], list[dict], dict]:
    accepted=[]; review=[]; report={'counts_by_type':Counter(),'status_counts':Counter(),'source_label_confusion':Counter(),'confidence_distribution':Counter(),'examples':defaultdict(list),'rejected':0,'invalid':0}
    seen_hash=set()
    for rec in rows:
        text=rec['text']; out_ents=[]; review_ents=[]
        for src in _source_entities(rec):
            start,end=int(src['start']),int(src['end']); source_label=str(src.get('source_label',''))
            if not (0 <= start < end <= len(text)) or text[start:end] != src.get('text', text[start:end]):
                report['invalid']+=1; continue
            decision=classify_candidate(text, source_label, start, end, qwen_enabled)
            base={'id':'','start':start,'end':end,'text':text[start:end],'type':decision.target if decision.target in TARGET_TYPES else 'IGNORE','assertions':[],'candidates':[], 'metadata':{'source_label':source_label,'weak_label_method':decision.method,'confidence':decision.confidence,'reason':decision.reason,'original_start':start,'original_end':end,'record_id':rec.get('id')}}
            if decision.target in TARGET_TYPES and decision.confidence >= min_confidence:
                if any(_overlaps(base, e) for e in out_ents): report['invalid']+=1; continue
                base['id']=f"E{len(out_ents)+1}"; out_ents.append(base); report['counts_by_type'][decision.target]+=1; report['status_counts']['accepted']+=1
                report['source_label_confusion'][(source_label,decision.target)]+=1
                if len(report['examples'][decision.target]) < 5: report['examples'][decision.target].append(base['text'])
            elif decision.target == 'REJECT' or decision.target == 'IGNORE':
                report['status_counts']['rejected']+=1; report['rejected']+=1
            else:
                base['metadata']['needs_review']=True; base['review_status']='pending'; review_ents.append(base); report['status_counts']['review']+=1
            bucket=str(round(decision.confidence,1)); report['confidence_distribution'][bucket]+=1
        if out_ents:
            out={'id':rec.get('id'), 'text':text, 'entities':out_ents, 'relations':[], 'source':rec.get('source','weak_external'), 'source_split':'train', 'license':rec.get('license','local'), 'metadata':{**rec.get('metadata',{}), 'synthetic':False, 'silver':True, 'upstream_id':rec.get('metadata',{}).get('upstream_id', rec.get('id'))}}
            validate_record(out); h=ann_hash(out)
            if h not in seen_hash: seen_hash.add(h); accepted.append(out)
        if review_ents:
            review.append({'id':rec.get('id'),'text':text,'review_status':'pending','reviewer':'','review_notes':'','source':rec.get('source','weak_external'),'source_split':'train','proposed_entities':review_ents,'metadata':{**rec.get('metadata',{}), 'silver':False, 'gold_evaluation':False}})
    rep={k:(dict(v) if isinstance(v, Counter) else dict(v) if isinstance(v, defaultdict) else v) for k,v in report.items()}
    rep['source_label_confusion']={f"{k[0]}->{k[1]}":v for k,v in report['source_label_confusion'].items()}
    vals=list(rep['counts_by_type'].values()); rep['class_imbalance_ratio']=(max(vals)/min(vals)) if vals and min(vals)>0 else None
    rep['duplicate_leakage']='checked_train_only_no_dev_test_inputs'
    return accepted, review, rep


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0: return []
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False)+'\n')


def build_weak_corpus(train_real: Path, synthetic_train: Path, out_dir: Path, annotation_dir: Path, min_confidence: float=0.9, qwen_enabled: bool=False) -> dict:
    rows=read_jsonl(train_real)
    accepted, review, report=weak_label_records(rows, min_confidence, qwen_enabled)
    synth=read_jsonl(synthetic_train)
    for r in synth: r.setdefault('metadata',{})['synthetic']=True
    write_jsonl(out_dir/'train.silver.jsonl', accepted)
    write_jsonl(annotation_dir/'train.silver.todo.jsonl', review)
    write_jsonl(out_dir/'train.external.jsonl', accepted)
    write_jsonl(out_dir/'train.combined.jsonl', accepted+synth)
    write_jsonl(annotation_dir/'gold_dev.todo.jsonl', [])
    write_jsonl(annotation_dir/'local_test.todo.jsonl', [])
    (out_dir/'weak_label_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
