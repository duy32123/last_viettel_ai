import json, subprocess, sys
from pathlib import Path
from src.data.build_kb import build
from src.data.import_icd import import_icd_csv
from src.data.kb_schema import dedupe_records


def test_importer_dedupes_versioned_codes():
    recs=import_icd_csv(Path("tests/fixtures/kb/icd_fixture.csv"),"ICD-10","fixture-2026","fixture", False)
    out=dedupe_records(recs)
    assert len(out)==1 and out[0].code=="I10" and "cao huyết áp" in out[0].synonyms


def test_build_kb_fixture(tmp_path):
    cfg=tmp_path/"cfg.yaml"; cfg.write_text('{"output_dir":"'+str(tmp_path)+'","sources":[{"name":"icd","kind":"icd_csv","terminology":"ICD-10","version":"v1","path":"tests/fixtures/kb/icd_fixture.csv","verified":false},{"name":"rx","kind":"rxnorm_csv","version":"v1","path":"tests/fixtures/kb/rxnorm_fixture.csv","verified":false}]}', encoding="utf-8")
    counts=build(cfg)
    assert counts=={"icd":1,"rxnorm":1}
    assert (tmp_path/"manifest.json").exists()
    assert not json.loads((tmp_path/"icd.jsonl").read_text(encoding="utf-8").splitlines()[0])["verified"]


def test_production_config_skips_without_user_data(tmp_path):
    cfg=tmp_path/"prod.json"; cfg.write_text('{"output_dir":"'+str(tmp_path)+'","sources":[{"name":"missing","kind":"icd_csv","terminology":"ICD-10","version":"user-provided","path":"'+str(tmp_path/'missing.csv')+'","verified":true}]}', encoding="utf-8")
    assert build(cfg) == {"icd":0,"rxnorm":0}
    assert json.loads((tmp_path/"manifest.json").read_text())[0]["status"] == "skipped_missing"


def test_legacy_seed_keeps_configured_icd_variant_for_dotted_and_plain_codes(tmp_path):
    subprocess.run([sys.executable,"scripts/export_seed_kb.py","--out-dir",str(tmp_path),"--icd-terminology","ICD-10"], check=True)
    rows=[json.loads(l) for l in (tmp_path/"icd10_seed.jsonl").read_text(encoding="utf-8").splitlines()]
    selected={r["code"]:r for r in rows if r["code"] in {"I10","E11.9"}}
    assert selected["I10"]["terminology"] == "ICD-10"
    assert selected["E11.9"]["terminology"] == "ICD-10"
    assert selected["E11.9"]["metadata"]["variant_unknown"] is True
