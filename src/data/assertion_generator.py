from __future__ import annotations
import hashlib, json, random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from src.models.assertion.labels import ASSERTION_LABELS, ordered

ENTITY_BANK={
 "TRIỆU_CHỨNG":["sốt","ho","đau ngực","khó thở","mệt mỏi"],
 "CHẨN_ĐOÁN":["tăng huyết áp","đái tháo đường","hen phế quản","viêm phổi","béo phì"],
 "TÊN_XÉT_NGHIỆM":["HbA1c","CRP","INR","glucose","WBC"],
 "KẾT_QUẢ_XÉT_NGHIỆM":["8.1%","20 mg/L","2.5","dương tính","cao"],
 "THUỐC":["insulin","amlodipine","metformin","paracetamol","ceftriaxone"],
}
ASSERTION_ORDER=ASSERTION_LABELS

@dataclass
class AssertionGenConfig:
    seed:int=83
    train_negated:int=600
    train_family:int=600
    train_historical:int=600
    train_none:int=1200
    combo_min:int=100
    dev_per_family:int=8
    test_per_family:int=8
    audit:int=120

TEMPLATE_FAMILIES={
 "negation":[("Không ghi nhận {m}.", ["isNegated"]),("Không {m} nhưng còn {other}.", ["isNegated"]),("Phủ nhận {m}; hiện tại theo dõi.", ["isNegated"])],
 "family":[("Mẹ bệnh nhân có {m}.", ["isFamily"]),("Tiền sử gia đình có {m}.", ["isFamily","isHistorical"]),("Bố từng điều trị {m}.", ["isFamily","isHistorical"])],
 "historical":[("Tiền sử {m}.", ["isHistorical"]),("Trước đây bệnh nhân từng {m}.", ["isHistorical"]),("Đã điều trị {m} trước nhập viện.", ["isHistorical"])],
 "none":[("Ghi nhận {m}.", []),("Bệnh nhân bị {m}.", []),("Hiện tại {m} ổn định.", []),("Theo dõi {m}.", []),("Không những {m} mà còn {other}.", [])],
 "combo_neg_hist":[("Tiền sử không ghi nhận {m}.", ["isNegated","isHistorical"]),("Trước đây phủ nhận {m}.", ["isNegated","isHistorical"])],
 "combo_family_neg":[("Mẹ bệnh nhân không có {m}.", ["isNegated","isFamily"]),("Người nhà phủ nhận {m}.", ["isNegated","isFamily"])],
 "combo_all":[("Tiền sử gia đình không ghi nhận {m}.", ["isNegated","isFamily","isHistorical"])],
}
SPLIT_FAMILIES={"train":["negation","family","historical","none","combo_neg_hist","combo_family_neg","combo_all"],"dev":["dev_scope"],"test":["test_scope"]}
DEV_TEMPLATES=[("Không sốt nhưng còn ho.", [("sốt","TRIỆU_CHỨNG",["isNegated"]),("ho","TRIỆU_CHỨNG",[])]),("Tiền sử hen phế quản. Hiện tại khó thở.", [("hen phế quản","CHẨN_ĐOÁN",["isHistorical"]),("khó thở","TRIỆU_CHỨNG",[])])]
TEST_TEMPLATES=[("Mẹ bệnh nhân có đái tháo đường.", [("đái tháo đường","CHẨN_ĐOÁN",["isFamily"])]),("Không những sốt mà còn ho.", [("sốt","TRIỆU_CHỨNG",[]),("ho","TRIỆU_CHỨNG",[])])]

def _entity(text, mention, typ, labels):
    s=text.index(mention); e=s+len(mention)
    return {"text":mention,"type":typ,"position":[s,e],"candidates":[],"assertions":ordered(labels)}

def _record(split, family, idx, text, entities):
    return {"id":f"assert_{split}_{family}_{idx}","text":text,"entities":entities,"source":"assertion_synthetic","source_split":split,"license":"project_generated","metadata":{"synthetic":True,"template_family":family,"gold_evaluation":split in {"dev","test"}}}

def _choose_entity(rng, i):
    typ=list(ENTITY_BANK)[i % len(ENTITY_BANK)]; return typ, rng.choice(ENTITY_BANK[typ])

def _render_train(cfg: AssertionGenConfig) -> list[dict[str,Any]]:
    rng=random.Random(cfg.seed); rows=[]; targets={"isNegated":cfg.train_negated,"isFamily":cfg.train_family,"isHistorical":cfg.train_historical,"NONE":cfg.train_none,"isFamily+isHistorical":cfg.combo_min,"isNegated+isHistorical":cfg.combo_min,"isNegated+isFamily":cfg.combo_min}
    counts=Counter(); i=0
    while any(counts[k] < v for k,v in targets.items()):
        for fam in SPLIT_FAMILIES["train"]:
            tmpl,labels=rng.choice(TEMPLATE_FAMILIES[fam]); typ,m=_choose_entity(rng,i); other="ho" if m!="ho" else "sốt"
            text=tmpl.format(m=m, other=other)
            text = f"Lượt khám {i}: " + text
            ent=_entity(text,m,typ,labels)
            # If the template includes an other mention, annotate it as NONE when it is a supported symptom.
            ents=[ent]
            if "{other}" in tmpl and other in text:
                ents.append(_entity(text, other, "TRIỆU_CHỨNG", []))
            rows.append(_record("train", fam, i, text, ents))
            for _e in ents:
                _labs=_e.get("assertions", [])
                labs="+".join(_labs) if _labs else "NONE"
                counts[labs]+=1
                for lab in _labs: counts[lab]+=1
                if not _labs: counts["NONE"]+=1
            i+=1
            if not any(counts[k] < v for k,v in targets.items()): break
    def _none_entities():
        return sum(1 for r in rows for e in r["entities"] if not e.get("assertions"))
    while _none_entities() < cfg.train_none:
        typ,m=_choose_entity(rng,i); text=f"Lượt khám {i}: Ghi nhận {m}."; ent=_entity(text,m,typ,[])
        rows.append(_record("train", "none", i, text, [ent])); i += 1
    return rows

def _render_eval(split, templates, n):
    rows=[]; k=0
    for i in range(n):
        base_text, specs=templates[i % len(templates)]; text=f"Phiên {i}: " + base_text; ents=[_entity(text,m,t,labs) for m,t,labs in specs]
        rows.append(_record(split, f"{split}_scope", k, text, ents)); k+=1
    return rows

def canonical_hash(rec):
    anns=sorted((e["position"][0],e["position"][1],e["type"],tuple(e.get("assertions",[]))) for e in rec["entities"])
    return hashlib.sha256(json.dumps([rec["text"],anns], ensure_ascii=False).encode()).hexdigest()

def validate_records(rows):
    for r in rows:
        for e in r["entities"]:
            s,en=e["position"]
            if not (0 <= s < en <= len(r["text"])) or r["text"][s:en] != e["text"]: raise ValueError(f"bad offset {r['id']} {e}")
            ordered(e.get("assertions", []))

def build_report(splits: dict[str,list[dict[str,Any]]]) -> dict[str,Any]:
    report={"record_counts":{k:len(v) for k,v in splits.items()},"label_positive_counts":{},"label_negative_counts":{},"combination_counts":{},"template_families":{},"duplicates":{},"leakage":False,"invalid_offsets":0,"audit_pending":0}
    all_hashes={}
    for split,rows in splits.items():
        pos=Counter(); neg=Counter(); combo=Counter(); fam=set(); seen=Counter()
        for r in rows:
            fam.add(r["metadata"]["template_family"]); seen[canonical_hash(r)] += 1
            for e in r["entities"]:
                labs=set(e.get("assertions", [])); combo["+".join(ordered(labs)) or "NONE"] += 1
                for lab in ASSERTION_LABELS:
                    (pos if lab in labs else neg)[lab] += 1
        report["label_positive_counts"][split]=dict(pos); report["label_negative_counts"][split]=dict(neg); report["combination_counts"][split]=dict(combo); report["template_families"][split]=sorted(fam); report["duplicates"][split]=sum(c-1 for c in seen.values() if c>1)
        for h in seen:
            if h in all_hashes and all_hashes[h] != split: report["leakage"]=True
            all_hashes[h]=split
    return report

def assert_gate(report, cfg: AssertionGenConfig):
    tr=report["label_positive_counts"]["train"]; combo=report["combination_counts"]["train"]
    for lab,minv in [("isNegated",cfg.train_negated),("isFamily",cfg.train_family),("isHistorical",cfg.train_historical)]:
        if tr.get(lab,0) < minv: raise ValueError(f"{lab} below minimum")
    if combo.get("NONE",0) < cfg.train_none: raise ValueError("NONE below minimum")
    for c in ["isFamily+isHistorical","isNegated+isHistorical","isNegated+isFamily"]:
        if combo.get(c,0) < cfg.combo_min: raise ValueError(f"{c} below minimum")
    if report["leakage"] or any(report["duplicates"].values()): raise ValueError("leakage/duplicates detected")

def build_assertion_corpus(out_dir: str|Path="data/processed/assertion", audit_path: str|Path="data/annotation/assertion_audit.todo.jsonl", cfg: AssertionGenConfig|None=None):
    cfg=cfg or AssertionGenConfig(); out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    splits={"train":_render_train(cfg),"dev":_render_eval("dev", DEV_TEMPLATES, cfg.dev_per_family),"test":_render_eval("test", TEST_TEMPLATES, cfg.test_per_family)}
    for rows in splits.values(): validate_records(rows)
    report=build_report(splits); assert_gate(report,cfg)
    for split,rows in splits.items(): (out/f"{split}.jsonl").write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in rows)+"\n", encoding="utf-8")
    audit=[]
    for r in splits["train"][:cfg.audit]:
        audit.append({"id":r["id"],"text":r["text"],"proposed_entities":r["entities"],"review_status":"pending","reviewer":"","review_notes":"","metadata":{"gold_evaluation":False}})
    ap=Path(audit_path); ap.parent.mkdir(parents=True, exist_ok=True); ap.write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in audit)+"\n", encoding="utf-8")
    report["audit_pending"]=len(audit); (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8")
    return report
