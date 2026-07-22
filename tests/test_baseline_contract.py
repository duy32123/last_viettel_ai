import json, subprocess, sys
from pathlib import Path
from extract import extract_concepts

VALID={"TRIỆU_CHỨNG","TÊN_XÉT_NGHIỆM","KẾT_QUẢ_XÉT_NGHIỆM","CHẨN_ĐOÁN","THUỐC"}

def check(text, concepts):
    assert concepts == sorted(concepts, key=lambda r:(r["position"][0], r["position"][1]))
    for c in concepts:
        assert c["type"] in VALID
        s,e=c["position"]
        assert text[s:e] == c["text"]

def test_unicode_crlf_lf_dose_assertions_repeats():
    text="1. Tiền sử bệnh\r\nMẹ bệnh nhân tăng huyết áp, dùng amlodipine 10 mg po daily.\nHiện tại: không đau ngực nhưng ho và ho. WBC: 12,5 /mm3"
    concepts=extract_concepts(text); check(text, concepts)
    assert any(c["type"]=="THUỐC" and "10 mg" in c["text"] for c in concepts)
    assert any("isFamily" in c.get("assertions",[]) for c in concepts)
    assert any("isHistorical" in c.get("assertions",[]) for c in concepts)
    assert any("isNegated" in c.get("assertions",[]) for c in concepts)
    assert sum(1 for c in concepts if c["text"].lower()=="ho") == 2

def test_cli_reads_sorted_txt_and_writes_json(tmp_path):
    inp=tmp_path/"input"; out=tmp_path/"output"; inp.mkdir()
    (inp/"2.txt").write_text("Bệnh nhân ho.", encoding="utf-8")
    (inp/"1.txt").write_text("WBC: 5 /mm3\nparacetamol 500 mg po", encoding="utf-8")
    cp=subprocess.run([sys.executable,"run_pipeline.py",str(inp),str(out)], text=True, capture_output=True, check=True, encoding="utf-8")
    assert "Đã xử lý 2 file" in cp.stdout
    assert [p.name for p in sorted(out.glob("*.json"))] == ["1.json","2.json"]
    for p in out.glob("*.json"):
        data=json.loads(p.read_text(encoding="utf-8")); check((inp/(p.stem+".txt")).read_text(encoding="utf-8"), data)
