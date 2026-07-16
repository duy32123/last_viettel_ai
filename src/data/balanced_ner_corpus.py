from __future__ import annotations
import json, random, hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from src.data.dataset_schema import VALID_TYPES, validate_record
from src.data.synthetic.generator import ann_hash

TARGET_TYPES=sorted(VALID_TYPES)
SOURCE_TARGET_CONSTRAINTS={
    "DRUGCHEMICAL":{"THUỐC","TÊN_XÉT_NGHIỆM","IGNORE"},
    "DISEASESYMTOM":{"TRIỆU_CHỨNG","CHẨN_ĐOÁN","IGNORE"},
    "DIAGNOSTICS":{"TÊN_XÉT_NGHIỆM","IGNORE"},
    "UNITCALIBRATOR":{"TÊN_XÉT_NGHIỆM","KẾT_QUẢ_XÉT_NGHIỆM","IGNORE"},
}

@dataclass
class BalanceConfig:
    min_entities_per_type: int = 500
    min_unique_mentions_per_type: int = 100
    min_unique_mentions_drug_diagnosis: int = 200
    max_class_imbalance_ratio: float = 3.0
    max_top_mention_share: float = 0.02
    audit_samples_per_type: int = 50
    seed: int = 57

SYMPTOMS=["sốt","ho khan","khó thở","đau ngực","mệt mỏi","đau bụng","buồn nôn","chóng mặt","đau đầu","khò khè","ớn lạnh","đau họng","sụt cân","phù chân","tiêu chảy","nôn ói","đau lưng","đau khớp","ngứa","phát ban"]
DIAGNOSES=["tăng huyết áp","đái tháo đường","béo phì","viêm phổi","hen phế quản","suy tim","bệnh thận mạn","viêm gan B","rối loạn lipid máu","gout","lao phổi","viêm dạ dày","thiếu máu","cường giáp","suy giáp","COPD","sỏi thận","viêm xoang","trầm cảm","loãng xương"]
DRUGS=["insulin","metformin","amlodipine","losartan","atorvastatin","aspirin","paracetamol","ibuprofen","omeprazole","cefuroxime","amoxicillin","azithromycin","salbutamol","budesonide","furosemide","spironolactone","warfarin","clopidogrel","zoloft","prozac","lisinopril","bisoprolol","glimepiride","sitagliptin","dapagliflozin"]
TESTS=["HbA1c","CRP","INR","glucose","WBC","AST","ALT","creatinine","hemoglobin","ure","troponin","LDL-C","HDL-C","triglyceride","bilirubin","albumin","natri","kali","procalcitonin","D-dimer"]
RESULTS=["7.2 mmol/L","20 mg/L","dương tính","âm tính","tăng","giảm","cao","thấp","5.8%","12 G/L","1.1 mg/dL","140 mmol/L","3.4 mmol/L","250 U/L","80 mg/dL"]
CONTEXTS=["Bệnh nhân ghi nhận {m}.","BN có {m}; cần theo dõi.","Không ghi nhận {m} trước đó.","Tiền sử gia đình có {m}.","Khám hôm nay: {m}!", "Triệu chứng: {m}\r\nĐề nghị tái khám."]
DRUG_CONTEXTS=["Đang dùng {m} 5 mg đường uống.","Kê {m} liều thấp sau ăn.","BN tự mua {m}, uống mỗi ngày.","Thuốc hiện tại: {m}; theo dõi tác dụng phụ."]
TEST_CONTEXTS=["Xét nghiệm {m} được chỉ định.","Theo dõi {m} buổi sáng.","Kết quả {m}: đang chờ.","XN {m}\r\nLặp lại sau 1 tuần."]
RESULT_CONTEXTS=["HbA1c {m}, tư vấn kiểm soát đường huyết.","CRP {m}; cân nhắc nhiễm trùng.","INR {m}, chỉnh liều nếu cần.","glucose {m}; đo lại khi đói.","WBC {m}, theo dõi công thức máu."]


def _mention(base: str, idx: int, typ: str) -> str:
    if typ == "THUỐC": return f"{base}-{idx:03d}"
    if typ == "CHẨN_ĐOÁN": return f"{base} type {idx:03d}"
    if typ == "TRIỆU_CHỨNG": return f"{base} mức {idx:03d}"
    if typ == "TÊN_XÉT_NGHIỆM": return f"{base}-{idx:03d}"
    return RESULTS[idx % len(RESULTS)] if idx < len(RESULTS) else f"{idx + 1}.{idx%10} mmol/L"


def _record(record_id: str, text: str, mention: str, typ: str, source: str="targeted_synthetic_v2") -> dict[str,Any]:
    start=text.index(mention); end=start+len(mention)
    rec={"id":record_id,"text":text,"entities":[{"id":"E1","start":start,"end":end,"text":mention,"type":typ,"assertions":[],"candidates":[],"metadata":{"targeted_synthetic":True}}],"relations":[],"source":source,"source_split":"train","license":"project-generated","metadata":{"synthetic":True,"targeted_synthetic":True,"gold_evaluation":False,"phase":"5F"}}
    validate_record(rec); return rec


def source_target_allowed(source_label: str, target: str) -> bool:
    return target in SOURCE_TARGET_CONSTRAINTS.get(source_label.upper(), {"IGNORE"})



def apply_source_target_constraints(rec: dict[str,Any]) -> dict[str,Any] | None:
    kept=[]
    for ent in rec.get("entities", []):
        source_label=ent.get("metadata", {}).get("source_label") or ent.get("source_label")
        if source_label and not source_target_allowed(str(source_label), ent.get("type", "IGNORE")):
            continue
        kept.append(ent)
    if rec.get("entities") and not kept:
        return None
    out={**rec, "entities": kept}
    return out

def generate_targeted_synthetic(cfg: BalanceConfig) -> list[dict[str,Any]]:
    rng=random.Random(cfg.seed); rows=[]; specs=[("TRIỆU_CHỨNG",SYMPTOMS,CONTEXTS),("CHẨN_ĐOÁN",DIAGNOSES,CONTEXTS),("THUỐC",DRUGS,DRUG_CONTEXTS),("TÊN_XÉT_NGHIỆM",TESTS,TEST_CONTEXTS)]
    for typ,bases,contexts in specs:
        for i in range(cfg.min_entities_per_type):
            base=bases[i % len(bases)]; mention=_mention(base, i, typ)
            ctx=contexts[i % len(contexts)]
            if i % 11 == 0: mention=mention.upper()
            if i % 17 == 0: mention=mention.replace(" ", "-")
            text=ctx.format(m=mention)
            rows.append(_record(f"v2_{typ}_{i:04d}", text, mention, typ))
    for i in range(cfg.min_entities_per_type):
        mention=_mention("", i, "KẾT_QUẢ_XÉT_NGHIỆM")
        text=RESULT_CONTEXTS[i % len(RESULT_CONTEXTS)].format(m=mention)
        rows.append(_record(f"v2_KQXN_{i:04d}", text, mention, "KẾT_QUẢ_XÉT_NGHIỆM"))
    rng.shuffle(rows)
    return rows


def _read_jsonl(path: Path) -> list[dict[str,Any]]:
    if not path.exists() or path.stat().st_size == 0: return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jsonl(path: Path, rows: list[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False)+"\n")


def _norm(s: str) -> str:
    return " ".join(s.casefold().strip().split())


def balanced_report(rows: list[dict[str,Any]]) -> dict[str,Any]:
    counts=Counter(); forms=defaultdict(Counter); sources=Counter(); dup=0; seen=set(); invalid=0
    for r in rows:
        sources[r.get("source","unknown")]+=1
        try: validate_record(r)
        except Exception: invalid+=1
        h=ann_hash(r)
        if h in seen: dup+=1
        seen.add(h)
        for e in r.get("entities",[]):
            if e.get("type") in VALID_TYPES:
                counts[e["type"]]+=1; forms[e["type"]][_norm(e["text"])] += 1
    top_share={t:(max(c.values())/sum(c.values()) if c else 0.0) for t,c in forms.items()}
    vals=[counts[t] for t in TARGET_TYPES if counts[t]>0]
    return {"entity_count_by_type":dict(counts),"unique_forms_by_type":{t:len(forms[t]) for t in TARGET_TYPES},"top_mention_share_by_type":top_share,"top_mentions_by_type":{t:dict(forms[t].most_common(10)) for t in TARGET_TYPES},"source_composition":dict(sources),"class_imbalance_ratio":(max(vals)/min(vals) if vals else None),"duplicate_records":dup,"invalid_offsets":invalid}


def assert_balanced_gate(report: dict[str,Any], cfg: BalanceConfig) -> None:
    errors=[]
    for typ in TARGET_TYPES:
        if report["entity_count_by_type"].get(typ,0) < cfg.min_entities_per_type: errors.append(f"{typ} entity minimum not met")
        min_unique=cfg.min_unique_mentions_drug_diagnosis if typ in {"THUỐC","CHẨN_ĐOÁN"} else cfg.min_unique_mentions_per_type
        if report["unique_forms_by_type"].get(typ,0) < min_unique: errors.append(f"{typ} unique-form minimum not met")
        if report["top_mention_share_by_type"].get(typ,1.0) > cfg.max_top_mention_share: errors.append(f"{typ} top mention share too high")
    if report.get("class_imbalance_ratio") and report["class_imbalance_ratio"] > cfg.max_class_imbalance_ratio: errors.append("class imbalance too high")
    if report.get("duplicate_records") or report.get("invalid_offsets"): errors.append("duplicate or invalid offsets found")
    if errors: raise ValueError("; ".join(errors))


def build_balanced_corpus(train_silver: Path, train_synthetic: Path, output_path: Path, audit_path: Path, report_path: Path, cfg: BalanceConfig) -> dict[str,Any]:
    # Inputs are train split only; dev/test/competition paths are intentionally not accepted.
    rows=[]; seen=set()
    for r in _read_jsonl(train_silver) + _read_jsonl(train_synthetic) + generate_targeted_synthetic(cfg):
        if r.get("source_split") != "train": continue
        r=apply_source_target_constraints(r)
        if r is None: continue
        r.setdefault("metadata",{})["phase5f_train_only"] = True
        h=ann_hash(r)
        if h not in seen:
            validate_record(r); seen.add(h); rows.append(r)
    report=balanced_report(rows); assert_balanced_gate(report, cfg)
    _write_jsonl(output_path, rows)
    audit=[]; per_type=Counter()
    for r in rows:
        typ=r["entities"][0]["type"] if r.get("entities") else None
        if typ in TARGET_TYPES and per_type[typ] < cfg.audit_samples_per_type:
            audit.append({"id":r["id"],"text":r["text"],"review_status":"pending","reviewer":"","review_notes":"","source":r["source"],"proposed_entities":r["entities"],"metadata":{"gold_evaluation":False,"silver":True,"phase":"5F"}}); per_type[typ]+=1
    _write_jsonl(audit_path, audit)
    report["audit_queue_by_type"]=dict(per_type); report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
