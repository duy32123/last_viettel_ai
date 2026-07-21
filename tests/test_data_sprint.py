import json
from pathlib import Path
import pytest
from src.data.adapters.span_jsonl import load_span_jsonl, DEFAULT_MAPPINGS
from src.data.corpus import build_corpus
from src.data.gold_dev import promote


def test_vimq_mapping_offsets_and_ambiguous_unmapped():
    rows, report = load_span_jsonl(Path("tests/fixtures/real_corpus/vimq_fixture.jsonl"), "ViMQ", "dev", "fixture", DEFAULT_MAPPINGS["vimq"])
    assert rows[0]["text"][rows[0]["entities"][0]["start"]:rows[0]["entities"][0]["end"]] == "Panadol"
    assert rows[0]["entities"][0]["type"] == "THUỐC"
    assert {e["type"] for e in rows[0]["entities"][1:]} == {"UNMAPPED"}
    assert report["source_labels"]["SYMPTOM_AND_DISEASE"] == 2
    assert "sốt" in report["unmapped_examples"]["SYMPTOM_AND_DISEASE"]


def test_vietmed_mapping_unicode_crlf_and_reconstruction(tmp_path):
    p=tmp_path/"vietmed.jsonl"
    text="Bệnh nhân ho khan\r\nWBC 7 G/L"
    p.write_text(json.dumps({"id":"crlf","text":text,"entities":[{"start":10,"end":17,"label":"SYMPTOM","text":"ho khan"},{"start":19,"end":22,"label":"TEST_NAME","text":"WBC"},{"start":23,"end":28,"label":"TEST_RESULT","text":"7 G/L"}]}, ensure_ascii=False)+"\n", encoding="utf-8")
    mapping={"SYMPTOM":"TRIỆU_CHỨNG","TEST_NAME":"TÊN_XÉT_NGHIỆM","TEST_RESULT":"KẾT_QUẢ_XÉT_NGHIỆM"}
    rows,_=load_span_jsonl(p,"VietMed-NER_MultiMed","train","fixture",mapping)
    assert [e["text"] for e in rows[0]["entities"]] == ["ho khan","WBC","7 G/L"]


def test_license_aware_missing_local_data_and_corpus_outputs(tmp_path):
    cfg=json.loads(Path("tests/fixtures/real_corpus/corpus_config.json").read_text(encoding="utf-8"))
    escaped_tmp_path=json.dumps(str(tmp_path))[1:-1]
    cfg_text=json.dumps(cfg, ensure_ascii=False).replace("OUT", escaped_tmp_path)
    cfg_path=tmp_path/"cfg.json"; cfg_path.write_text(cfg_text, encoding="utf-8")
    report=build_corpus(cfg_path)
    assert report["real_records"] == 4
    assert report["synthetic_records"] > 0
    assert report["mapping_counts"]["unmapped"] >= 2
    assert (tmp_path/"processed/train.real.jsonl").exists()
    assert (tmp_path/"processed/train.combined.jsonl").exists()
    assert (tmp_path/"processed/dev.provisional.jsonl").exists()
    todo=tmp_path/"annotation/gold_dev.todo.jsonl"
    assert todo.exists()
    first=json.loads(todo.read_text(encoding="utf-8").splitlines()[0])
    assert first["review_status"] == "pending" and "proposed_entities" in first
    manifest=json.loads((tmp_path/"processed/real_corpus_manifest.json").read_text(encoding="utf-8"))
    assert all(s["status"] == "loaded_local" for s in manifest["sources"])


def test_corpus_blocks_upstream_leakage(tmp_path):
    dup=tmp_path/"dup.jsonl"
    dup.write_text(json.dumps({"id":"same","text":"Panadol","entities":[{"start":0,"end":7,"label":"DRUG","text":"Panadol"}]}, ensure_ascii=False)+"\n", encoding="utf-8")
    cfg={"output_dir":str(tmp_path/"p"),"annotation_dir":str(tmp_path/"a"),"datasets":[
        {"name":"ViMQ","adapter":"vimq","local_path":str(dup),"split":"train","license_terms":"fixture","redistribution_allowed":True},
        {"name":"ViMQ","adapter":"vimq","local_path":str(dup),"split":"dev","license_terms":"fixture","redistribution_allowed":True}
    ]}
    cfgp=tmp_path/"cfg.json"; cfgp.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="upstream document leakage"):
        build_corpus(cfgp)


def test_gold_promotion_only_approved_and_valid(tmp_path):
    todo=tmp_path/"todo.jsonl"; out=tmp_path/"dev.gold.jsonl"
    approved={"id":"r1","text":"sốt","source":"review","source_split":"dev","license":"fixture","metadata":{},"proposed_entities":[{"id":"E1","start":0,"end":3,"text":"sốt","type":"TRIỆU_CHỨNG","assertions":[],"candidates":[]}],"review_status":"approved","reviewer":"human","review_notes":"ok"}
    pending={**approved,"id":"r2","review_status":"pending"}
    todo.write_text(json.dumps(approved, ensure_ascii=False)+"\n"+json.dumps(pending, ensure_ascii=False)+"\n", encoding="utf-8")
    res=promote(todo,out)
    assert res["approved"] == 1 and res["skipped"] == 1
    row=json.loads(out.read_text(encoding="utf-8"))
    assert row["metadata"]["gold_evaluation"] is True