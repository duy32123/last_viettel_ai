from __future__ import annotations
import json, random, re, unicodedata
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
FAKE_ID_PATTERNS=[re.compile(r"-\d{3}$"), re.compile(r"\btype \d{3}\b", re.I), re.compile(r"\bmức \d{3}\b", re.I)]

@dataclass
class BalanceConfig:
    min_entities_per_type: int = 500
    min_canonical_concepts_per_type: int = 15
    min_canonical_concepts_drug_diagnosis: int = 20
    max_class_imbalance_ratio: float = 3.0
    max_top_canonical_share: float = 0.07
    audit_samples_per_type: int = 50
    seed: int = 57
    profile: str = "experimental"

SYMPTOM_INV=[("sym_fever",["sốt","sot","sốt cao"]),("sym_dry_cough",["ho khan","ho khan nhiều"]),("sym_dyspnea",["khó thở","kho tho"]),("sym_chest_pain",["đau ngực","dau nguc"]),("sym_fatigue",["mệt mỏi","mệt"]),("sym_abdominal_pain",["đau bụng"]),("sym_nausea",["buồn nôn"]),("sym_dizziness",["chóng mặt"]),("sym_headache",["đau đầu"]),("sym_wheeze",["khò khè"]),("sym_chills",["ớn lạnh"]),("sym_sore_throat",["đau họng"]),("sym_weight_loss",["sụt cân"]),("sym_edema",["phù chân"]),("sym_diarrhea",["tiêu chảy"])]
DIAG_INV=[("dx_htn",["tăng huyết áp","THA"]),("dx_diabetes",["đái tháo đường","tiểu đường"]),("dx_obesity",["béo phì"]),("dx_pneumonia",["viêm phổi"]),("dx_asthma",["hen phế quản","hen"]),("dx_hf",["suy tim"]),("dx_ckd",["bệnh thận mạn","BTM"]),("dx_hepb",["viêm gan B"]),("dx_dyslipidemia",["rối loạn lipid máu"]),("dx_gout",["gout"]),("dx_tb",["lao phổi"]),("dx_gastritis",["viêm dạ dày"]),("dx_anemia",["thiếu máu"]),("dx_hyperthyroid",["cường giáp"]),("dx_hypothyroid",["suy giáp"]),("dx_copd",["COPD"]),("dx_stone",["sỏi thận"]),("dx_sinusitis",["viêm xoang"]),("dx_depression",["trầm cảm"]),("dx_osteoporosis",["loãng xương"])]
DRUG_INV=[("drug_insulin",["insulin"]),("drug_metformin",["metformin"]),("drug_amlodipine",["amlodipine"]),("drug_losartan",["losartan"]),("drug_atorvastatin",["atorvastatin"]),("drug_aspirin",["aspirin"]),("drug_paracetamol",["paracetamol","acetaminophen"]),("drug_ibuprofen",["ibuprofen"]),("drug_omeprazole",["omeprazole"]),("drug_cefuroxime",["cefuroxime"]),("drug_amoxicillin",["amoxicillin"]),("drug_azithro",["azithromycin"]),("drug_salbutamol",["salbutamol"]),("drug_budesonide",["budesonide"]),("drug_furosemide",["furosemide"]),("drug_spironolactone",["spironolactone"]),("drug_warfarin",["warfarin"]),("drug_clopidogrel",["clopidogrel"]),("drug_zoloft",["Zoloft","sertraline"]),("drug_prozac",["Prozac","fluoxetine"])]
TEST_INV=[("test_hba1c",["HbA1c","HBA1C"]),("test_crp",["CRP"]),("test_inr",["INR"]),("test_glucose",["glucose","đường huyết"]),("test_wbc",["WBC","bạch cầu"]),("test_ast",["AST"]),("test_alt",["ALT"]),("test_creatinine",["creatinine"]),("test_hb",["hemoglobin","Hb"]),("test_ure",["ure"]),("test_troponin",["troponin"]),("test_ldl",["LDL-C"]),("test_hdl",["HDL-C"]),("test_tg",["triglyceride"]),("test_bilirubin",["bilirubin"]),("test_sodium",["natri","Na+"]),("test_potassium",["kali","K+"]),("test_pct",["procalcitonin"]),("test_ddimer",["D-dimer"]),("test_albumin",["albumin"]),("test_ferritin",["ferritin"]),("test_tsh",["TSH"]),("test_ldh",["LDH"]),("test_sarscov2",["SARS-CoV-2"]),("test_flua",["cúm A"]),("test_esr",["ESR"])]
RESULT_INV=[("result_hba1c_pct",["7.2%","8.1%","5.8%"]),("result_crp_mgl",["20 mg/L","35 mg/L"]),("result_inr",["2.5","3.1"]),("result_glucose_mmol",["7.2 mmol/L","8.1 mmol/L"]),("result_wbc_gl",["12 G/L","15 G/L"]),("result_positive",["dương tính"]),("result_negative",["âm tính"]),("result_high",["cao"]),("result_low",["thấp"]),("result_increased",["tăng"]),("result_decreased",["giảm"]),("result_creatinine",["1.1 mg/dL","1.4 mg/dL"]),("result_ast",["80 U/L"]),("result_alt",["95 U/L"]),("result_hb",["10 g/dL"]),("result_sodium",["140 mmol/L"]),("result_potassium",["3.4 mmol/L"]),("result_pct",["0.8 ng/mL"]),("result_ddimer",["500 ng/mL"]),("result_albumin",["35 g/L"])]
INVENTORY={"TRIỆU_CHỨNG":SYMPTOM_INV,"CHẨN_ĐOÁN":DIAG_INV,"THUỐC":DRUG_INV,"TÊN_XÉT_NGHIỆM":TEST_INV,"KẾT_QUẢ_XÉT_NGHIỆM":RESULT_INV}
SYMPTOM_CONTEXTS=["Bệnh nhân than {m}.","BN có biểu hiện {m}; cần theo dõi.","Không ghi nhận {m} trong lần khám này.","Khám hôm nay ghi nhận {m}!", "Triệu chứng: {m}\r\nĐề nghị tái khám."]
DIAGNOSIS_CONTEXTS=["Chẩn đoán hiện tại: {m}.","Tiền sử bệnh: {m}.","Bệnh nền gồm {m}; theo dõi định kỳ.","Hồ sơ ghi nhận {m} đã điều trị.","Kết luận sau khám: {m}."]
DRUG_CONTEXTS=["Đang dùng {m} 5 mg đường uống.","Kê {m} liều thấp sau ăn.","BN tự mua {m}, uống mỗi ngày.","Thuốc hiện tại: {m}; theo dõi tác dụng phụ."]
TEST_CONTEXTS=["Xét nghiệm {m} được chỉ định.","Theo dõi {m} buổi sáng.","Kết quả {m}: đang chờ.","XN {m}\r\nLặp lại sau 1 tuần."]
RESULT_CONTEXTS_BY_CONCEPT={
    "result_hba1c_pct":["HbA1c {m}, tư vấn kiểm soát đường huyết.","Kết quả HbA1c là {m} sau 3 tháng."],
    "result_crp_mgl":["CRP {m}; cân nhắc nhiễm trùng.","Xét nghiệm CRP cho thấy {m}."],
    "result_inr":["INR {m}, chỉnh liều nếu cần.","Kết quả INR là {m}."],
    "result_glucose_mmol":["glucose {m}; đo lại khi đói.","Đường huyết glucose ghi nhận {m}."],
    "result_wbc_gl":["WBC {m}, theo dõi công thức máu.","Bạch cầu WBC ở mức {m}."],
    "result_positive":["SARS-CoV-2 {m}.","Kháng nguyên SARS-CoV-2 {m}."],
    "result_negative":["cúm A {m}.","Test nhanh cúm A {m}."],
    "result_high":["ferritin {m} so với ngưỡng tham chiếu.","ferritin {m} sau ăn."],
    "result_low":["TSH {m} so với ngưỡng tham chiếu.","TSH {m} lúc đói."],
    "result_increased":["LDH {m} so với lần trước.","LDH {m} sau điều trị."],
    "result_decreased":["ESR {m} sau kháng sinh.","ESR {m} sau điều chỉnh thuốc."],
    "result_creatinine":["creatinine {m}, đánh giá chức năng thận.","Kết quả creatinine là {m}."],
    "result_ast":["AST {m}, theo dõi men gan.","Men gan AST ghi nhận {m}."],
    "result_alt":["ALT {m}, theo dõi men gan.","Men gan ALT ghi nhận {m}."],
    "result_hb":["Hb {m}, đánh giá thiếu máu.","hemoglobin {m} trong công thức máu."],
    "result_sodium":["natri {m}, theo dõi điện giải.","natri {m} trong ion đồ."],
    "result_potassium":["kali {m}, theo dõi điện giải.","kali {m} trong ion đồ."],
    "result_pct":["procalcitonin {m}, đánh giá nhiễm trùng.","procalcitonin {m} sau xét nghiệm."],
    "result_ddimer":["D-dimer {m}, theo dõi đông máu.","Kết quả D-dimer là {m}."],
    "result_albumin":["albumin {m}, đánh giá dinh dưỡng.","Kết quả albumin là {m}."],
}
TEMPLATE_BY_TYPE={"TRIỆU_CHỨNG":SYMPTOM_CONTEXTS,"CHẨN_ĐOÁN":DIAGNOSIS_CONTEXTS,"THUỐC":DRUG_CONTEXTS,"TÊN_XÉT_NGHIỆM":TEST_CONTEXTS}


def canonical_surface(text: str) -> str:
    text=unicodedata.normalize("NFC", text).casefold()
    text=re.sub(r"\d+(?:[.,]\d+)?", "<NUM>", text)
    text=re.sub(r"[^\w<>/%]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _entity(text: str, mention: str, typ: str, concept_id: str, eid: int) -> dict[str,Any]:
    start=text.index(mention); end=start+len(mention)
    if any(p.search(mention) for p in FAKE_ID_PATTERNS): raise ValueError(f"fake synthetic identifier in mention: {mention}")
    return {"id":f"E{eid}","start":start,"end":end,"text":mention,"type":typ,"assertions":[],"candidates":[],"metadata":{"targeted_synthetic":True,"concept_id":concept_id,"canonical_surface":canonical_surface(mention)}}

def _record_with_entities(record_id: str, text: str, entities: list[dict[str,Any]], source: str="targeted_synthetic_v2", context_template_id: str="") -> dict[str,Any]:
    rec={"id":record_id,"text":text,"entities":sorted(entities, key=lambda e:e["start"]),"relations":[],"source":source,"source_split":"train","license":"project-generated","metadata":{"synthetic":True,"targeted_synthetic":True,"gold_evaluation":False,"phase":"5G","context_template_id":context_template_id}}
    validate_record(rec); return rec

def _record(record_id: str, text: str, mention: str, typ: str, concept_id: str, source: str="targeted_synthetic_v2", context_template_id: str="") -> dict[str,Any]:
    return _record_with_entities(record_id, text, [_entity(text, mention, typ, concept_id, 1)], source, context_template_id)

def _lab_pair_record(record_id: str, text: str, test_name: str, result_value: str, test_concept_id: str, result_concept_id: str, context_template_id: str="") -> dict[str,Any]:
    if test_name in result_value or any(name in result_value.split() for name,_ in []): raise ValueError("test name leaked into result span")
    ents=[_entity(text, test_name, "TÊN_XÉT_NGHIỆM", test_concept_id, 1), _entity(text, result_value, "KẾT_QUẢ_XÉT_NGHIỆM", result_concept_id, 2)]
    if max(ents[0]["start"], ents[1]["start"]) < min(ents[0]["end"], ents[1]["end"]): raise ValueError("overlapping lab pair")
    return _record_with_entities(record_id, text, ents, context_template_id=context_template_id)


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
    return {**rec, "entities": kept}


RESULT_TEST_CONCEPT={
    "result_hba1c_pct":("test_hba1c","HbA1c"),"result_crp_mgl":("test_crp","CRP"),"result_inr":("test_inr","INR"),"result_glucose_mmol":("test_glucose","glucose"),"result_wbc_gl":("test_wbc","WBC"),"result_positive":("test_sarscov2","SARS-CoV-2"),"result_negative":("test_flua","cúm A"),"result_high":("test_ferritin","ferritin"),"result_low":("test_tsh","TSH"),"result_increased":("test_ldh","LDH"),"result_decreased":("test_esr","ESR"),"result_creatinine":("test_creatinine","creatinine"),"result_ast":("test_ast","AST"),"result_alt":("test_alt","ALT"),"result_hb":("test_hb","hemoglobin"),"result_sodium":("test_sodium","natri"),"result_potassium":("test_potassium","kali"),"result_pct":("test_pct","procalcitonin"),"result_ddimer":("test_ddimer","D-dimer"),"result_albumin":("test_albumin","albumin"),
}

def _render_result_context(concept_id: str, mention: str, i: int) -> tuple[str,str,str,int]:
    test_concept_id, test_name=RESULT_TEST_CONCEPT[concept_id]
    contexts=RESULT_CONTEXTS_BY_CONCEPT[concept_id]
    template_idx=(i + len(concept_id)) % len(contexts)
    text=contexts[template_idx].format(m=mention) + f" Lần khám {i}."
    return text, test_name, test_concept_id, template_idx

def _multi_pair_records(start_idx: int=0, copies: int=5) -> list[dict[str,Any]]:
    rows=[]
    for i in range(copies):
        text=("Xét nghiệm HbA1c 8.1%, CRP 20 mg/L và INR 2.5." if i % 2 == 0 else "XN HbA1c 7.2%\r\nCRP 35 mg/L; INR 3.1.") + f" Lô {i}."
        pairs=[("HbA1c","8.1%" if i % 2 == 0 else "7.2%","test_hba1c","result_hba1c_pct"),("CRP","20 mg/L" if i % 2 == 0 else "35 mg/L","test_crp","result_crp_mgl"),("INR","2.5" if i % 2 == 0 else "3.1","test_inr","result_inr")]
        ents=[]
        for n,(test_name,result_value,test_cid,result_cid) in enumerate(pairs):
            ents.append(_entity(text,test_name,"TÊN_XÉT_NGHIỆM",test_cid,2*n+1)); ents.append(_entity(text,result_value,"KẾT_QUẢ_XÉT_NGHIỆM",result_cid,2*n+2))
        rows.append(_record_with_entities(f"v2_1_multi_lab_{start_idx+i:04d}", text, ents, context_template_id=f"multi_lab:{i%2}"))
    return rows

def generate_targeted_synthetic(cfg: BalanceConfig) -> list[dict[str,Any]]:
    rng=random.Random(cfg.seed); rows=[]
    for typ, concepts in INVENTORY.items():
        target_n=100 if typ == "TÊN_XÉT_NGHIỆM" else cfg.min_entities_per_type
        if typ == "KẾT_QUẢ_XÉT_NGHIỆM":
            for i in range(cfg.min_entities_per_type):
                concept_id, surfaces=concepts[i % len(concepts)]
                mention=surfaces[(i // len(concepts)) % len(surfaces)]
                text,test_name,test_concept_id,template_idx=_render_result_context(concept_id, mention, i)
                rows.append(_lab_pair_record(f"v2_1_lab_pair_{i:04d}", text, test_name, mention, test_concept_id, concept_id, context_template_id=f"KẾT_QUẢ_XÉT_NGHIỆM:{concept_id}:{template_idx}"))
            continue
        contexts=TEMPLATE_BY_TYPE[typ]
        for i in range(target_n):
            concept_id, surfaces=concepts[i % len(concepts)]
            mention=surfaces[(i // len(concepts)) % len(surfaces)]
            template_idx=(i + len(concept_id)) % len(contexts)
            text=contexts[template_idx].format(m=mention) + f" Lần khám {i}."
            rows.append(_record(f"v2_{typ}_{i:04d}", text, mention, typ, concept_id, context_template_id=f"{typ}:{concept_id}:{template_idx}"))
    rows.extend(_multi_pair_records(copies=5))
    rng.shuffle(rows)
    return rows


def _read_jsonl(path: Path) -> list[dict[str,Any]]:
    if not path.exists() or path.stat().st_size == 0: return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jsonl(path: Path, rows: list[dict[str,Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False)+"\n")


def _source_bucket(rec: dict[str,Any]) -> str:
    if rec.get("metadata", {}).get("silver"): return "real_silver"
    if rec.get("source") == "targeted_synthetic_v2": return "targeted_synthetic"
    if rec.get("metadata", {}).get("synthetic") or str(rec.get("source", "")).startswith("synthetic"): return "existing_synthetic"
    return "other_train_input"


def _concept_id(ent: dict[str,Any]) -> str:
    return ent.get("metadata", {}).get("concept_id") or canonical_surface(ent.get("text", ""))


def balanced_report(rows: list[dict[str,Any]]) -> dict[str,Any]:
    counts=Counter(); raw_forms=defaultdict(Counter); concepts=defaultdict(Counter); sources=Counter(); templates=defaultdict(set); paired_lab=0; missing_lab_name=0; dup=0; seen=set(); invalid=0
    for r in rows:
        sources[_source_bucket(r)]+=1
        try: validate_record(r)
        except Exception: invalid+=1
        h=ann_hash(r)
        if h in seen: dup+=1
        seen.add(h)
        types_in_record={e.get("type") for e in r.get("entities", [])}
        if "KẾT_QUẢ_XÉT_NGHIỆM" in types_in_record:
            if "TÊN_XÉT_NGHIỆM" in types_in_record: paired_lab += 1
            else: missing_lab_name += 1
        for e in r.get("entities",[]):
            if e.get("type") in VALID_TYPES:
                typ=e["type"]; counts[typ]+=1; raw_forms[typ][unicodedata.normalize("NFC", e["text"]).casefold()] += 1; concepts[typ][_concept_id(e)] += 1; templates[typ].add(r.get("metadata", {}).get("context_template_id") or re.sub(r"\bLần khám \d+\.", "Lần khám {n}.", r["text"].replace(e["text"], "{m}")))
    top_share={t:(max(c.values())/sum(c.values()) if c else 0.0) for t,c in concepts.items()}
    vals=[counts[t] for t in TARGET_TYPES if counts[t]>0]
    return {"entity_count_by_type":dict(counts),"raw_surface_forms":{t:len(raw_forms[t]) for t in TARGET_TYPES},"canonical_concepts":{t:len(concepts[t]) for t in TARGET_TYPES},"context_template_count":{t:len(templates[t]) for t in TARGET_TYPES},"top_canonical_share_by_type":top_share,"top_concepts_by_type":{t:dict(concepts[t].most_common(10)) for t in TARGET_TYPES},"source_composition":dict(sources),"paired_lab_result_records":paired_lab,"lab_result_records_missing_test_name":missing_lab_name,"class_imbalance_ratio":(max(vals)/min(vals) if vals else None),"duplicate_records":dup,"invalid_offsets":invalid}


def assert_balanced_gate(report: dict[str,Any], cfg: BalanceConfig) -> None:
    errors=[]
    for typ in TARGET_TYPES:
        if report["entity_count_by_type"].get(typ,0) < cfg.min_entities_per_type: errors.append(f"{typ} entity minimum not met")
        min_concepts=cfg.min_canonical_concepts_drug_diagnosis if typ in {"THUỐC","CHẨN_ĐOÁN"} else cfg.min_canonical_concepts_per_type
        if report["canonical_concepts"].get(typ,0) < min_concepts: errors.append(f"{typ} canonical-concept minimum not met")
        if report["top_canonical_share_by_type"].get(typ,1.0) > cfg.max_top_canonical_share: errors.append(f"{typ} top canonical share too high")
    if report.get("class_imbalance_ratio") and report["class_imbalance_ratio"] > cfg.max_class_imbalance_ratio: errors.append("class imbalance too high")
    if report.get("duplicate_records") or report.get("invalid_offsets"): errors.append("duplicate or invalid offsets found")
    if errors: raise ValueError("; ".join(errors))


def build_balanced_corpus(train_silver: Path, train_synthetic: Path, output_path: Path, audit_path: Path, report_path: Path, cfg: BalanceConfig) -> dict[str,Any]:
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
        types={e["type"] for e in r.get("entities", []) if e.get("type") in TARGET_TYPES}
        needed={t for t in types if per_type[t] < cfg.audit_samples_per_type}
        if needed:
            audit.append({"id":r["id"],"text":r["text"],"review_status":"pending","reviewer":"","review_notes":"","source":r["source"],"proposed_entities":r["entities"],"metadata":{"gold_evaluation":False,"silver":True,"phase":"5G"}})
            for typ in needed: per_type[typ]+=1
    _write_jsonl(audit_path, audit)
    report["audit_queue_by_type"]=dict(per_type); report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
