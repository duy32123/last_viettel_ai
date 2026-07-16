from pathlib import Path
from src.data.build_kb import build
from src.data.import_icd import import_icd_csv
from src.data.kb_schema import dedupe_records

def test_importer_dedupes_versioned_codes():
    recs=import_icd_csv(Path("tests/fixtures/kb/icd_fixture.csv"),"ICD-10","fixture-2026","fixture")
    out=dedupe_records(recs)
    assert len(out)==1 and out[0].code=="I10" and "cao huyết áp" in out[0].synonyms

def test_build_kb_fixture(tmp_path):
    cfg=tmp_path/"cfg.yaml"; cfg.write_text('{"output_dir":"'+str(tmp_path)+'","sources":[{"name":"icd","kind":"icd_csv","terminology":"ICD-10","version":"v1","path":"tests/fixtures/kb/icd_fixture.csv","verified":true},{"name":"rx","kind":"rxnorm_csv","version":"v1","path":"tests/fixtures/kb/rxnorm_fixture.csv","verified":true}]}', encoding="utf-8")
    counts=build(cfg)
    assert counts=={"icd":1,"rxnorm":1}
    assert (tmp_path/"manifest.json").exists()
