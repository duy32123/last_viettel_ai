import json
from pathlib import Path
from src.data.adapters.simple_jsonl import load_jsonl
from src.data.synthetic.generator import generate, assert_no_leakage

def test_dataset_fixture_adapter():
    rows=load_jsonl(Path("tests/fixtures/datasets/phoner_fixture.jsonl"),"PhoNER_COVID19","train","fixture")
    assert rows[0]["entities"][0]["text"] == "sốt"

def test_synthetic_offsets_determinism_and_leakage(tmp_path):
    stats1=generate(13,{"train":4,"dev":2,"test":2},tmp_path)
    first=(tmp_path/"train.jsonl").read_text(encoding="utf-8")
    stats2=generate(13,{"train":4,"dev":2,"test":2},tmp_path)
    assert first==(tmp_path/"train.jsonl").read_text(encoding="utf-8")
    assert stats1==stats2
    for file in ["train.jsonl","dev.jsonl","test.jsonl"]:
        for line in (tmp_path/file).read_text(encoding="utf-8").splitlines():
            r=json.loads(line)
            for e in r["entities"]: assert r["text"][e["start"]:e["end"]] == e["text"]
    assert_no_leakage({"train":tmp_path/"train.jsonl","dev":tmp_path/"dev.jsonl","test":tmp_path/"test.jsonl"})
