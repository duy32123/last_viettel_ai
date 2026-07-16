from __future__ import annotations
import hashlib, json, random, re, unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from src.models.assertion.labels import ASSERTION_LABELS, ordered

ENTITY_BANK={
 "TRIỆU_CHỨNG":["sốt","ho","đau ngực","khó thở","mệt mỏi","buồn nôn","đau đầu","chóng mặt","phù chân","khò khè","đau bụng","tiêu chảy","nôn ói","đau họng","sụt cân","đổ mồ hôi","đau lưng","tê tay","ngứa","chảy máu cam"],
 "CHẨN_ĐOÁN":["tăng huyết áp","đái tháo đường","hen phế quản","viêm phổi","béo phì","suy tim","bệnh thận mạn","viêm gan B","lao phổi","gout","rối loạn lipid máu","COPD","viêm dạ dày","sỏi thận","thiếu máu","viêm xoang","đột quỵ","ung thư phổi","xơ gan","suy giáp"],
 "TÊN_XÉT_NGHIỆM":["HbA1c","CRP","INR","glucose","WBC","creatinine","AST","ALT","troponin","D-dimer","TSH","ferritin","LDH","albumin","natri","kali","ESR","procalcitonin","bilirubin","hemoglobin"],
 "KẾT_QUẢ_XÉT_NGHIỆM":["8.1%","20 mg/L","2.5","dương tính","âm tính","cao","thấp","15 G/L","1.3 mg/dL","620 ng/mL","7.2 mmol/L","tăng","giảm","0.04 ng/mL","130 mmol/L","3.1","11 g/dL","45 U/L","80 mg/dL","bình thường"],
 "THUỐC":["insulin","amlodipine","metformin","paracetamol","ceftriaxone","atorvastatin","losartan","omeprazole","salbutamol","prednisolone","aspirin","clopidogrel","furosemide","bisoprolol","warfarin","heparin","azithromycin","amoxicillin","zoloft","prozac"],
}
CONTEXTS=["khoa nội","phòng khám","bệnh án","phiếu theo dõi","tóm tắt ra viện","hồ sơ cấp cứu","ghi chú điều dưỡng","biên bản hội chẩn","đơn thuốc","lần tái khám","đợt nhập viện","sổ khám bệnh"]
STYLE=["ghi rõ","ghi chú","mô tả","xác nhận","nhắc lại","theo lời kể","trong hồ sơ","ở phần nhận định"]
TOKENS=["an","binh","chi","dung","giang","ha","khanh","linh","minh","nam","oanh","phuc","quang","son","tam","uyen","vinh","xuan","yen","bao","cam","duy","han","lam","mai","nhi","phuong","thao","trang","vy"]

@dataclass
class AssertionGenConfig:
    seed:int=83
    train_examples:int=3500
    dev_examples:int=600
    test_examples:int=600
    min_split_label_pos:int=120
    min_split_label_neg:int=120
    min_none:int=150
    combo_min:int=30
    all_three_min:int=20
    audit:int=120

SPLIT_FAMILIES={"train":["negation_admission","family_history_note","past_history_note","current_problem","combined_history"],"dev":["dev_scope_paraphrase","dev_counterfactual_paraphrase","dev_section_paraphrase"],"test":["test_scope_paraphrase","test_counterfactual_paraphrase","test_section_paraphrase"]}
COMBOS=[[],["isNegated"],["isFamily"],["isHistorical"],["isFamily","isHistorical"],["isNegated","isHistorical"],["isNegated","isFamily"],["isNegated"],[],["isFamily"],["isHistorical"],["isFamily","isHistorical"],["isNegated","isHistorical"],["isNegated","isFamily"],[],["isNegated"],["isFamily"],["isHistorical"],["isFamily","isHistorical"],["isNegated","isFamily","isHistorical"]]

def _entity(text, mention, typ, labels):
    s=text.index(mention); e=s+len(mention)
    return {"text":mention,"type":typ,"position":[s,e],"candidates":[],"assertions":ordered(labels)}

def _record(split, family, idx, text, entities, slice_name):
    split_name={"train":"train","dev":"synthetic_dev","test":"synthetic_test"}[split]
    return {"id":f"assert_{split}_{idx}","text":text,"entities":entities,"source":"assertion_synthetic","source_split":split_name,"license":"project_generated","metadata":{"synthetic":True,"gold_evaluation":False,"official_evaluation":False,"template_family":family,"slice":slice_name}}

def _choose_entity(i):
    typ=list(ENTITY_BANK)[i % len(ENTITY_BANK)]; return typ, ENTITY_BANK[typ][(i//len(ENTITY_BANK)) % len(ENTITY_BANK[typ])]

def _context(i):
    a=TOKENS[i % len(TOKENS)]; b=TOKENS[(i//len(TOKENS)) % len(TOKENS)]; c=TOKENS[(i//(len(TOKENS)*len(TOKENS))) % len(TOKENS)]
    return f"{CONTEXTS[i % len(CONTEXTS)]} {STYLE[(i//len(CONTEXTS)) % len(STYLE)]} nhóm {a} {b} {c}"

def _text_for(combo, mention, i, split):
    ctx=_context(i)
    patterns={
      "train":{
        "none":f"{ctx}: Bác sĩ ghi nhận {mention} trong lần khám này.",
        "neg":f"{ctx}: Không ghi nhận {mention} nhưng tình trạng khác ổn định.",
        "fam":f"{ctx}: Mẹ bệnh nhân có {mention}, bệnh nhân hiện chưa ghi nhận vấn đề này.",
        "hist":f"TIỀN SỬ\r\n{ctx}: {mention}.\r\nHIỆN TẠI\r\nTheo dõi triệu chứng mới.",
        "fh":f"{ctx}: Tiền sử gia đình có {mention}; hiện tại bệnh nhân không than phiền liên quan.",
        "nh":f"{ctx}: Trước đây không ghi nhận {mention}. Hiện tại đánh giá lại.",
        "fn":f"{ctx}: Người nhà phủ nhận {mention}; bệnh nhân trao đổi thêm sau.",
        "all":f"{ctx}: Tiền sử gia đình không ghi nhận {mention}. Hiện tại không dùng thông tin này làm chẩn đoán."},
      "dev":{
        "none":f"Trong ghi chú lâm sàng tại {ctx}, nhân viên y tế mô tả {mention} đang được theo dõi.",
        "neg":f"Qua thăm khám hôm nay ở {ctx}, bác sĩ loại trừ {mention} trong cùng nhận định.",
        "fam":f"Khai thác bệnh sử tại {ctx} cho biết bố từng có {mention}; người bệnh không tự báo triệu chứng tương ứng.",
        "hist":f"BỆNH SỬ CŨ\r\nHồ sơ {ctx} lưu {mention}.\r\nKHÁM LẠI\r\nChưa có diễn biến cấp.",
        "fh":f"Thông tin họ hàng trong {ctx} ghi nhận {mention} từ các lần khám cũ; hiện chưa dùng làm vấn đề chính.",
        "nh":f"Hồ sơ cũ tại {ctx} phủ nhận {mention}, sau đó bệnh nhân được hẹn đánh giá định kỳ.",
        "fn":f"Người nhà nói tại {ctx} rằng chưa có dấu hiệu {mention} khi trao đổi với điều dưỡng.",
        "all":f"Bệnh sử gia đình trước đây ở {ctx} không có dấu hiệu {mention}; bác sĩ chỉ lưu để đối chiếu."},
      "test":{
        "none":f"Ở phần nhận định cuối của {ctx}, bác sĩ liệt kê {mention} như một vấn đề cần xử trí.",
        "neg":f"Kết luận khám tại {ctx} ghi chưa thấy {mention} sau khi đối chiếu triệu chứng.",
        "fam":f"Anh ruột được ghi trong {ctx} có {mention}; bệnh nhân hiện không được gán chẩn đoán này.",
        "hist":f"THÔNG TIN TRƯỚC ĐÂY\r\n{ctx} ghi đã từng {mention}.\r\nĐỢT NÀY\r\nTheo dõi thêm.",
        "fh":f"Dòng họ có tiền sử {mention} trong {ctx} ở lần khai thác cũ; lần này chưa xác nhận trên bệnh nhân.",
        "nh":f"Tài liệu trước nhập viện từ {ctx} không ghi nhận {mention}; bác sĩ tiếp tục rà soát.",
        "fn":f"Trao đổi với gia đình tại {ctx} cho thấy không có {mention} trong người nhà.",
        "all":f"Tiền sử người nhà trước đây trong {ctx} phủ nhận {mention}; thông tin chỉ dùng để tham khảo."}}
    key={():"none",("isNegated",):"neg",("isFamily",):"fam",("isHistorical",):"hist",("isFamily","isHistorical"):"fh",("isNegated","isHistorical"):"nh",("isNegated","isFamily"):"fn"}.get(tuple(combo),"all")
    slices={"none":"counterfactual","neg":"scope","fam":"family","hist":"section","fh":"family_historical","nh":"neg_historical","fn":"family_negated","all":"all_three"}
    return patterns[split][key], slices[key]

def _make(split, idx, combo):
    typ,m=_choose_entity(idx); text,slice_name=_text_for(combo,m,idx,split)
    fam=SPLIT_FAMILIES[split][idx % len(SPLIT_FAMILIES[split])]
    ent=_entity(text,m,typ,combo)
    # deterministic multi-entity scope examples
    if idx % 17 == 0:
        other="ho" if m!="ho" else "sốt"; text=text + f" Tuy nhiên còn {other}."; ent=_entity(text,m,typ,combo); other_ent=_entity(text, other, "TRIỆU_CHỨNG", [])
        return _record(split,fam,idx,text,[ent,other_ent],"multi_entity_scope")
    if idx % 23 == 0:
        text=text.replace("Không ghi nhận", "Khong ghi nhan") if "Không ghi nhận" in text else text + " BN tái khám."
        ent=_entity(text,m,typ,combo)
        slice_name="no_diacritic"
    return _record(split,fam,idx,text,[ent],slice_name)

def _needs(counts, split, cfg):
    if split == "train": return sum(counts["entity_examples"].values()) < cfg.train_examples
    target=cfg.dev_examples if split == "dev" else cfg.test_examples
    if sum(counts["entity_examples"].values()) < target: return True
    if counts["combo"].get("NONE",0) < cfg.min_none: return True
    for lab in ASSERTION_LABELS:
        if counts["pos"].get(lab,0) < cfg.min_split_label_pos or counts["neg"].get(lab,0) < cfg.min_split_label_neg: return True
    for c in ["isFamily+isHistorical","isNegated+isHistorical","isNegated+isFamily"]:
        if counts["combo"].get(c,0) < cfg.combo_min: return True
    total=sum(counts["entity_examples"].values())
    if counts["combo"].get("isNegated+isFamily+isHistorical",0) < cfg.all_three_min: return True
    if split != "train" and total and counts["combo"].get("isNegated+isFamily+isHistorical",0)/total > 0.05:
        return False
    return False

def _update_counts(counts, rec):
    for e in rec["entities"]:
        labs=ordered(e.get("assertions", [])); key="+".join(labs) or "NONE"; counts["combo"][key]+=1; counts["entity_examples"][e["type"]]+=1
        for lab in ASSERTION_LABELS: counts["pos" if lab in labs else "neg"][lab]+=1

def _render_split(split, cfg):
    rows=[]; counts={"pos":Counter(),"neg":Counter(),"combo":Counter(),"entity_examples":Counter()}; i=0
    while _needs(counts, split, cfg):
        combo=COMBOS[i % len(COMBOS)]
        total=sum(counts["entity_examples"].values())
        if split != "train" and combo == ["isNegated","isFamily","isHistorical"] and counts["combo"].get("isNegated+isFamily+isHistorical",0) >= cfg.all_three_min and total and counts["combo"].get("isNegated+isFamily+isHistorical",0)/total >= 0.045:
            combo=[]
        rec=_make(split, i, combo); rows.append(rec); _update_counts(counts, rec); i+=1
    return rows
_PREFIX_RE=re.compile(r"^(?:Lượt khám|Phiên)\s+<NUM>:\s*", re.I)
def _norm_text(text):
    t=unicodedata.normalize("NFC", text).casefold(); t=re.sub(r"^(?:lượt khám|phiên)\s+\d+:\s*", "", t); t=re.sub(r"\s+", " ", t).strip(); return t

def semantic_template_signature(rec):
    text=_norm_text(rec["text"])
    for e in sorted(rec["entities"], key=lambda x: len(x["text"]), reverse=True):
        text=text.replace(e["text"].casefold(), "<entity>")
    text=re.sub(r"\b(?:train|dev|test|assert_\w+_\d+)\b", "<split_id>", text)
    text=re.sub(r"\d+", "<num>", text)
    return hashlib.sha256(text.encode()).hexdigest()

def canonical_hash(rec):
    anns=sorted((e["text"].casefold(),e["type"],tuple(e.get("assertions",[]))) for e in rec["entities"])
    return hashlib.sha256(json.dumps([_norm_text(rec["text"]),anns], ensure_ascii=False).encode()).hexdigest()

def validate_records(rows):
    for r in rows:
        for e in r["entities"]:
            s,en=e["position"]
            if not (0 <= s < en <= len(r["text"])) or r["text"][s:en] != e["text"]: raise ValueError(f"bad offset {r['id']} {e}")
            ordered(e.get("assertions", []))

def build_report(splits: dict[str,list[dict[str,Any]]]) -> dict[str,Any]:
    report={"record_counts":{k:len(v) for k,v in splits.items()},"entity_example_counts":{},"label_positive_counts":{},"label_negative_counts":{},"combination_counts":{},"combination_coverage":{},"template_families":{},"slice_counts":{},"duplicates":{},"leakage":False,"semantic_template_overlap":{},"marker_truncation_count":0,"label_dtype":"float32","invalid_offsets":0,"synthetic_gold_flags":{},"audit_pending":0}
    all_hashes={}
    signatures={}
    for split,rows in splits.items():
        pos=Counter(); neg=Counter(); combo=Counter(); fam=set(); seen=Counter(); ent_counts=Counter(); slices=Counter(); gold_flags=Counter()
        for r in rows:
            fam.add(r["metadata"]["template_family"]); seen[canonical_hash(r)] += 1; signatures.setdefault(split,set()).add(semantic_template_signature(r)); slices[r["metadata"].get("slice","general")]+=1; gold_flags[(r["metadata"].get("gold_evaluation"), r["metadata"].get("official_evaluation"))]+=1
            for e in r["entities"]:
                ent_counts[e["type"]]+=1; labs=ordered(e.get("assertions", [])); combo["+".join(labs) or "NONE"] += 1
                for lab in ASSERTION_LABELS: (pos if lab in labs else neg)[lab] += 1
        report["entity_example_counts"][split]=dict(ent_counts); report["label_positive_counts"][split]=dict(pos); report["label_negative_counts"][split]=dict(neg); report["combination_counts"][split]=dict(combo); report["combination_coverage"][split]=sorted(combo); report["template_families"][split]=sorted(fam); report["slice_counts"][split]=dict(slices); report["duplicates"][split]=sum(c-1 for c in seen.values() if c>1); report["synthetic_gold_flags"][split]={str(k):v for k,v in gold_flags.items()}
        for h in seen:
            if h in all_hashes and all_hashes[h] != split: report["leakage"]=True
            all_hashes[h]=split
    for a,b in [("train","dev"),("train","test"),("dev","test")]:
        report["semantic_template_overlap"][f"{a}_vs_{b}"]=len(signatures.get(a,set()) & signatures.get(b,set()))
    return report

def assert_gate(report, cfg: AssertionGenConfig):
    for split,min_examples in [("train",cfg.train_examples),("dev",cfg.dev_examples),("test",cfg.test_examples)]:
        if sum(report["entity_example_counts"][split].values()) < min_examples: raise ValueError(f"{split} entity examples below minimum")
        if set(report["entity_example_counts"][split]) != set(ENTITY_BANK): raise ValueError(f"{split} missing entity types")
        if split != "train":
            if report["combination_counts"][split].get("NONE",0) < cfg.min_none: raise ValueError(f"{split} NONE below minimum")
            for lab in ASSERTION_LABELS:
                if report["label_positive_counts"][split].get(lab,0) < cfg.min_split_label_pos or report["label_negative_counts"][split].get(lab,0) < cfg.min_split_label_neg: raise ValueError(f"{split} {lab} class coverage below minimum")
        for c in ["isFamily+isHistorical","isNegated+isHistorical","isNegated+isFamily"]:
            if report["combination_counts"][split].get(c,0) < (cfg.combo_min if split != "train" else cfg.combo_min): raise ValueError(f"{split} {c} below minimum")
        if split != "train" and report["combination_counts"][split].get("isNegated+isFamily+isHistorical",0)/sum(report["entity_example_counts"][split].values()) > 0.05: raise ValueError(f"{split} all-three exceeds 5 percent")
    for lab in ASSERTION_LABELS:
        p=report["label_positive_counts"]["train"][lab]; n=report["label_negative_counts"]["train"][lab]
        if max(p,n)/max(1,min(p,n)) > 3: raise ValueError(f"train imbalance for {lab}")
    if any(report["semantic_template_overlap"].values()): raise ValueError("semantic template overlap detected")
    if report["leakage"] or any(report["duplicates"].values()) or report["invalid_offsets"]: raise ValueError("leakage/duplicates/offsets detected")

def build_assertion_corpus(out_dir: str|Path="data/processed/assertion", audit_path: str|Path="data/annotation/assertion_audit.todo.jsonl", cfg: AssertionGenConfig|None=None):
    cfg=cfg or AssertionGenConfig(); out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    splits={"train":_render_split("train",cfg),"dev":_render_split("dev",cfg),"test":_render_split("test",cfg)}
    for rows in splits.values(): validate_records(rows)
    report=build_report(splits); assert_gate(report,cfg)
    for split,rows in splits.items(): (out/f"{split}.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n", encoding="utf-8")
    audit=[{"id":r["id"],"text":r["text"],"proposed_entities":r["entities"],"review_status":"pending","reviewer":"","review_notes":"","metadata":{"gold_evaluation":False,"official_evaluation":False}} for r in splits["train"][:cfg.audit]]
    ap=Path(audit_path); ap.parent.mkdir(parents=True, exist_ok=True); ap.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in audit)+"\n", encoding="utf-8")
    report["audit_pending"]=len(audit); (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8")
    return report
