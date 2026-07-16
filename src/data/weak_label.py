from __future__ import annotations
import json, re, hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from src.data.dataset_schema import validate_record, VALID_TYPES
from src.data.synthetic.generator import ann_hash

DRUG_ALLOW={"prozac","zoloft","amlodipine","panadol","paracetamol","ibuprofen","metformin","insulin"}
DRUG_IGNORE={"vi khuẩn","ruồi giấm","drosophila","bacteria"}
DIAG_ALLOW={"béo phì","tăng huyết áp","đái tháo đường","obesity","hypertension","diabetes"}
SYMPTOM_ALLOW={"sốt","ho","ho khan","khó thở","đau ngực","mệt","đau bụng"}
IGNORE_TERMS={"vi khuẩn","ruồi giấm","drosophila","bacteria"}
TEST_NAMES={"glucose","wbc","inr","hba1c","hemoglobin","creatinine","crp","ast","alt"}
RESULT_WORDS={"cao","thấp","dương tính","âm tính","tăng","giảm","positive","negative","high","low"}
TARGET_TYPES=set(VALID_TYPES)
QWEN_LABELS=TARGET_TYPES | {"IGNORE"}
QWEN_PROMPT_VERSIONS=("phase5d-weak-label-v1-a", "phase5d-weak-label-v1-b")
DEFAULT_QWEN_MODEL="Qwen/Qwen3-4B-Instruct-2507"
NUMERIC_RESULT_RE=re.compile(r"^[<>]?\d+(?:[.,]\d+)?\s*(?:%|mmol/l|mg/dl|g/l|u/l|ng/ml|pg/ml|/mm3|x10\^?9/l)$", re.I)

@dataclass
class WeakDecision:
    target: str
    confidence: float
    method: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

class QwenBackend(Protocol):
    model_name: str
    def classify_batch(self, candidates: list[dict[str, Any]], prompt_version: str) -> list[str]: ...

class TransformersQwenBackend:
    def __init__(self, model_name: str=DEFAULT_QWEN_MODEL, device: str="auto", load_in_4bit: bool=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name=model_name; self.device=device
        kwargs={"device_map": device if device != "cpu" else None}
        if load_in_4bit: kwargs["load_in_4bit"]=True
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.model=AutoModelForCausalLM.from_pretrained(model_name, **{k:v for k,v in kwargs.items() if v is not None})
        if device == "cpu": self.model.to("cpu")
        self.model.eval()

    def classify_batch(self, candidates: list[dict[str, Any]], prompt_version: str) -> list[str]:
        labels=[]
        for cand in candidates:
            prompt=(
                f"Prompt {prompt_version}. Classify ONLY this exact medical NER span. "
                "Return one label exactly from: TRIỆU_CHỨNG, CHẨN_ĐOÁN, TÊN_XÉT_NGHIỆM, "
                "KẾT_QUẢ_XÉT_NGHIỆM, THUỐC, IGNORE. Do not change offsets or text.\n"
                f"Context: {cand['context']}\nMention: {cand['mention']}\nSource label: {cand['source_label']}\nLabel:"
            )
            inputs=self.tokenizer(prompt, return_tensors="pt")
            dev=next(self.model.parameters()).device
            inputs={k:v.to(dev) for k,v in inputs.items()}
            out=self.model.generate(**inputs, max_new_tokens=16, do_sample=False)
            gen=self.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            labels.append(_parse_qwen_label(gen))
        return labels

def _parse_qwen_label(text: str) -> str:
    t=text.strip().upper()
    for label in sorted(QWEN_LABELS, key=len, reverse=True):
        if label.upper() in t:
            return label
    return "IGNORE"

def context_window(text: str, start: int, end: int, radius: int=80) -> str:
    return text[max(0,start-radius):min(len(text), end+radius)]

def _norm(s: str) -> str:
    return " ".join(s.casefold().strip().split())

def _valid_span(text: str, start: int, end: int, mention: str|None=None) -> bool:
    return 0 <= start < end <= len(text) and (mention is None or text[start:end] == mention)

def classify_candidate(text: str, source_label: str, start: int, end: int, qwen_enabled: bool=False) -> WeakDecision:
    if not _valid_span(text, start, end):
        return WeakDecision("REJECT",0.0,"rule_reject","invalid_offset")
    mention=text[start:end]
    if not mention.strip(): return WeakDecision("REJECT",0.0,"rule_reject","empty_mention")
    m=_norm(mention); ctx=_norm(context_window(text,start,end))
    label=source_label.upper()
    if label in {"UNITCALIBRATOR","DIAGNOSTICS","DRUGCHEMICAL"} and m in TEST_NAMES:
        return WeakDecision("TÊN_XÉT_NGHIỆM",0.97,"rule_test_name_priority","known_test_name")
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
        looks_result=bool(NUMERIC_RESULT_RE.match(m)) or m in RESULT_WORDS
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

def _qwen_cache_key(cand: dict[str, Any], prompt_version: str, model_name: str) -> str:
    payload={k:cand[k] for k in ('text','mention','start','end','source_label')}
    payload['prompt_version']=prompt_version; payload['model']=model_name
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def _load_cache(path: Path|None) -> dict[str,str]:
    if not path or not path.exists(): return {}
    return json.loads(path.read_text(encoding='utf-8'))

def _save_cache(path: Path|None, cache: dict[str,str]) -> None:
    if not path: return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')

def _qwen_decision(cand: dict[str,Any], backend: QwenBackend, cache: dict[str,str], cache_path: Path|None) -> WeakDecision:
    if not _valid_span(cand['text'], cand['start'], cand['end'], cand['mention']):
        return WeakDecision("REJECT",0.0,"qwen_reject","invalid_offset_after_qwen")
    labels=[]; model_name=getattr(backend, 'model_name', DEFAULT_QWEN_MODEL)
    for prompt_version in QWEN_PROMPT_VERSIONS:
        key=_qwen_cache_key(cand, prompt_version, model_name)
        if key not in cache:
            label=backend.classify_batch([cand], prompt_version)[0]
            cache[key]=label if label in QWEN_LABELS else "IGNORE"
            _save_cache(cache_path, cache)
        labels.append(cache[key])
    agree=labels[0] == labels[1]
    meta={'qwen_model':model_name,'qwen_prompt_versions':list(QWEN_PROMPT_VERSIONS),'qwen_passes':labels,'qwen_agreement':agree,'provenance':'silver_qwen_two_pass'}
    if not agree:
        return WeakDecision("REVIEW",0.50,"qwen_disagreement","two_pass_disagreement",meta)
    if labels[0] == "IGNORE":
        return WeakDecision("IGNORE",0.90,"qwen_two_pass","qwen_ignore",meta)
    return WeakDecision(labels[0],0.90,"qwen_two_pass","two_pass_agreement",meta)

def weak_label_records(rows: list[dict[str,Any]], min_confidence: float=0.9, qwen_enabled: bool=False, qwen_backend: QwenBackend|None=None, qwen_cache: Path|None=None, qwen_max_candidates: int|None=None) -> tuple[list[dict], list[dict], dict]:
    accepted=[]; review=[]
    report={'counts_by_type':Counter(),'entity_count_by_type':Counter(),'record_count_by_type':Counter(),'status_counts':Counter(),'source_label_confusion':Counter(),'confidence_distribution':Counter(),'method_confidence':Counter(),'qwen_agreement':Counter(),'unique_surface_forms_by_type':defaultdict(Counter),'top_mentions':defaultdict(Counter),'examples':defaultdict(list),'rejected':0,'invalid':0}
    seen_hash=set(); qwen_used=0; cache=_load_cache(qwen_cache)
    for rec in rows:
        text=rec['text']; out_ents=[]; review_ents=[]; rec_types=set()
        for src in _source_entities(rec):
            start,end=int(src['start']),int(src['end']); source_label=str(src.get('source_label',''))
            if not _valid_span(text, start, end, src.get('text', text[start:end])):
                report['invalid']+=1; continue
            decision=classify_candidate(text, source_label, start, end, qwen_enabled)
            if decision.target == 'REVIEW' and qwen_enabled and qwen_backend and (qwen_max_candidates is None or qwen_max_candidates <= 0 or qwen_used < qwen_max_candidates):
                cand={'text':text,'mention':text[start:end],'start':start,'end':end,'source_label':source_label,'context':context_window(text,start,end)}
                decision=_qwen_decision(cand, qwen_backend, cache, qwen_cache); qwen_used+=1
            meta={'source_label':source_label,'weak_label_method':decision.method,'confidence':decision.confidence,'reason':decision.reason,'original_start':start,'original_end':end,'record_id':rec.get('id'), **decision.metadata}
            base={'id':'','start':start,'end':end,'text':text[start:end],'type':decision.target if decision.target in TARGET_TYPES else 'IGNORE','assertions':[],'candidates':[], 'metadata':meta}
            if decision.target in TARGET_TYPES and decision.confidence >= min_confidence:
                if any(_overlaps(base, e) for e in out_ents): report['invalid']+=1; continue
                base['id']=f"E{len(out_ents)+1}"; out_ents.append(base); rec_types.add(decision.target)
                report['counts_by_type'][decision.target]+=1; report['entity_count_by_type'][decision.target]+=1; report['status_counts']['accepted']+=1
                report['source_label_confusion'][(source_label,decision.target)]+=1
                surf=_norm(base['text']); report['unique_surface_forms_by_type'][decision.target][surf]+=1; report['top_mentions'][decision.target][surf]+=1
                if len(report['examples'][decision.target]) < 5: report['examples'][decision.target].append(base['text'])
            elif decision.target == 'REJECT' or decision.target == 'IGNORE':
                report['status_counts']['rejected']+=1; report['rejected']+=1
            else:
                base['metadata']['needs_review']=True; base['review_status']='pending'; review_ents.append(base); report['status_counts']['review']+=1
            bucket=str(round(decision.confidence,1)); report['confidence_distribution'][bucket]+=1; report['method_confidence'][(decision.method,bucket)]+=1
            if 'qwen_agreement' in decision.metadata: report['qwen_agreement']['agree' if decision.metadata['qwen_agreement'] else 'disagree']+=1
        if out_ents:
            out={'id':rec.get('id'), 'text':text, 'entities':out_ents, 'relations':[], 'source':rec.get('source','weak_external'), 'source_split':'train', 'license':rec.get('license','local'), 'metadata':{**rec.get('metadata',{}), 'synthetic':False, 'silver':True, 'upstream_id':rec.get('metadata',{}).get('upstream_id', rec.get('id'))}}
            validate_record(out); h=ann_hash(out)
            if h not in seen_hash:
                seen_hash.add(h); accepted.append(out)
                for t in rec_types: report['record_count_by_type'][t]+=1
        if review_ents:
            review.append({'id':rec.get('id'),'text':text,'review_status':'pending','reviewer':'','review_notes':'','source':rec.get('source','weak_external'),'source_split':'train','proposed_entities':review_ents,'metadata':{**rec.get('metadata',{}), 'silver':False, 'gold_evaluation':False}})
    _save_cache(qwen_cache, cache)
    rep={}
    for k,v in report.items():
        if isinstance(v, Counter):
            rep[k]=dict(v)
        elif isinstance(v, defaultdict):
            rep[k]={kk:(dict(vv.most_common(20)) if isinstance(vv, Counter) else vv) for kk,vv in v.items()}
        else:
            rep[k]=v
    rep['source_label_confusion']={f"{k[0]}->{k[1]}":v for k,v in report['source_label_confusion'].items()}
    rep['method_confidence']={f"{k[0]}@{k[1]}":v for k,v in report['method_confidence'].items()}
    rep['top_mentions']={k:dict(v.most_common(10)) for k,v in report['top_mentions'].items()}
    rep['unique_surface_forms_by_type']={k:len(v) for k,v in report['unique_surface_forms_by_type'].items()}
    vals=list(rep['counts_by_type'].values()); rep['class_imbalance_ratio']=(max(vals)/min(vals)) if vals and min(vals)>0 else None
    rep['invalid_offsets']=report['invalid']; rep['duplicate_leakage']='checked_train_only_no_dev_test_inputs'
    return accepted, review, rep

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0: return []
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False)+'\n')

def build_weak_corpus(train_real: Path, synthetic_train: Path, out_dir: Path, annotation_dir: Path, min_confidence: float=0.9, qwen_enabled: bool=False, qwen_backend: QwenBackend|None=None, qwen_cache: Path|None=None, qwen_max_candidates: int|None=None) -> dict:
    rows=read_jsonl(train_real)
    accepted, review, report=weak_label_records(rows, min_confidence, qwen_enabled, qwen_backend, qwen_cache, qwen_max_candidates)
    synth=read_jsonl(synthetic_train)
    for r in synth: r.setdefault('metadata',{})['synthetic']=True
    write_jsonl(out_dir/'train.silver.jsonl', accepted)
    write_jsonl(annotation_dir/'train.silver.todo.jsonl', review)
    write_jsonl(out_dir/'train.external.jsonl', accepted)
    write_jsonl(out_dir/'train.combined.jsonl', accepted+synth)
    (out_dir/'weak_label_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
