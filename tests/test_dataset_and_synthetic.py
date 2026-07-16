import json
import pytest
from pathlib import Path
from src.data.adapters.simple_jsonl import load_jsonl
from src.data.adapters.phoner_covid19 import load_native
from src.data.synthetic.generator import generate, assert_no_leakage, sample, TEMPLATE_FAMILIES
from src.data.dataset_schema import validate_record, serialize_competition_candidates


def entity_set(rec):
    return {(e["text"], e["type"]) for e in rec["entities"]}


def test_dataset_fixture_adapter():
    rows=load_jsonl(Path("tests/fixtures/datasets/phoner_fixture.jsonl"),"PhoNER_COVID19","train","fixture")
    assert rows[0]["entities"][0]["text"] == "sốt"


def test_phoner_native_mapping_uses_config_mapping():
    rows=load_native(Path("tests/fixtures/datasets/phoner_native.txt"), "PhoNER_COVID19", "train", "fixture", {"SYMPTOM_AND_DISEASE":"UNMAPPED"})
    assert rows[0]["entities"][0]["type"] == "UNMAPPED"
    assert rows[0]["entities"][0]["text"] == "ho khan"


def test_synthetic_offsets_determinism_seed_effect_and_leakage(tmp_path):
    kb={"diagnosis":["tests/fixtures/kb/icd_fixture_as_jsonl_missing"], "drug":[]}
    stats1=generate(13,{"train":4,"dev":2,"test":2},tmp_path, kb)
    first=(tmp_path/"train.jsonl").read_text(encoding="utf-8")
    stats2=generate(13,{"train":4,"dev":2,"test":2},tmp_path, kb)
    assert first==(tmp_path/"train.jsonl").read_text(encoding="utf-8")
    assert stats1==stats2
    generate(14,{"train":4,"dev":2,"test":2},tmp_path, kb)
    assert first != (tmp_path/"train.jsonl").read_text(encoding="utf-8")
    fams={}
    for file in ["train.jsonl","dev.jsonl","test.jsonl"]:
        fams[file]=set()
        for line in (tmp_path/file).read_text(encoding="utf-8").splitlines():
            r=json.loads(line); fams[file].add(r["metadata"]["template_family"])
            for e in r["entities"]: assert r["text"][e["start"]:e["end"]] == e["text"]
    assert fams["train.jsonl"].isdisjoint(fams["dev.jsonl"])
    assert fams["train.jsonl"].isdisjoint(fams["test.jsonl"])
    assert fams["dev.jsonl"].isdisjoint(fams["test.jsonl"])
    assert_no_leakage({"train":tmp_path/"train.jsonl","dev":tmp_path/"dev.jsonl","test":tmp_path/"test.jsonl"})


def test_synthetic_template_expected_entities_incomplete_labels():
    expected={
        "current_symptoms": {("khó thở","TRIỆU_CHỨNG"),("đau ngực","TRIỆU_CHỨNG"),("mệt","TRIỆU_CHỨNG")},
        "history_drug": {("tăng huyết áp","CHẨN_ĐOÁN"),("amlodipine","THUỐC")},
        "family_labs": {("đái tháo đường","CHẨN_ĐOÁN"),("sốt","TRIỆU_CHỨNG"),("WBC","TÊN_XÉT_NGHIỆM"),("12,5 /mm3","KẾT_QUẢ_XÉT_NGHIỆM")},
        "resp_labs": {("ho khan","TRIỆU_CHỨNG"),("sốt","TRIỆU_CHỨNG"),("glucose","TÊN_XÉT_NGHIỆM"),("7,2 mmol/l","KẾT_QUẢ_XÉT_NGHIỆM")},
        "mixed_followup": {("mệt","TRIỆU_CHỨNG"),("sốt","TRIỆU_CHỨNG"),("đái tháo đường","CHẨN_ĐOÁN")},
        "rx_change": {("amlodipine","THUỐC"),("ho khan","TRIỆU_CHỨNG")},
    }
    for i,family in enumerate(TEMPLATE_FAMILIES):
        assert entity_set(sample(i, family)) == expected[family]


def test_candidate_schema_and_competition_serializer():
    base={"text":"abc","entities":[{"id":"E1","start":0,"end":3,"text":"abc","type":"CHẨN_ĐOÁN","assertions":[],"candidates":[]}],"relations":[]}
    base["entities"][0]["candidates"]=["I10"]
    validate_record(base); assert serialize_competition_candidates(base["entities"][0]) == ["I10"]
    base["entities"][0]["candidates"]=[{"code":"I10","terminology":"ICD-10","verified":False}]
    validate_record(base); assert serialize_competition_candidates(base["entities"][0]) == ["I10"]
    base["entities"][0]["candidates"]=[{"code":"I10","terminology":"BAD","verified":False}]
    with pytest.raises(ValueError): validate_record(base)
