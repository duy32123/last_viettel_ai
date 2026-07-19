from src.data.import_hf_icd_aux import import_auxiliary_rows, make_pilot_examples, valid_icd10_code, deterministic_split_id
from src.linking.retrieval import LexicalIndex

def rows():
    return [
        {"id":"a","code":"E11.9","language":"vi","diagnosis":"Đái tháo đường type 2","journal_note":"Bệnh nhân đái tháo đường type 2 đang theo dõi."},
        {"id":"b","code":"I10","language":"vi","diagnosis":"Tăng huyết áp","journal_note":"Ghi nhận tăng huyết áp."},
        {"id":"b","code":"I10","language":"vi","diagnosis":"Tăng huyết áp","journal_note":"Ghi nhận tăng huyết áp."},
        {"id":"c","code":"BAD","language":"en","diagnosis":"bad","journal_note":"bad"},
    ]

def test_icd10_format_auxiliary_schema_not_verified_and_report():
    assert valid_icd10_code("E11.9") and valid_icd10_code("I10") and not valid_icd10_code("BAD")
    recs, report=import_auxiliary_rows(rows(), version="pilot")
    assert report["source_kind"] == "auxiliary_hf" and report["official_kb"] is False and report["synthetic_notes"] is True
    assert report["license"] == "CC-BY-4.0" and report["verified"] == 0 and report["full_icd10_coverage"] is False
    assert report["records"] == 4 and report["unique_codes"] == 2 and report["vietnamese_records"] == 3
    assert report["invalid_code_formats"] == 1 and report["duplicate_aliases"] >= 1
    assert all(r.verified is False for r in recs)
    e11=next(r for r in recs if r.code == "E11.9")
    assert any(p["raw_alias"].startswith("Bệnh nhân") for p in e11.metadata["alias_provenance"])

def test_auxiliary_alias_retrieval_and_deterministic_pilot_split():
    recs, _=import_auxiliary_rows(rows(), version="pilot")
    idx=LexicalIndex(recs, include_unverified=True)
    hit=idx.search("đái tháo đường type 2", "CHẨN_ĐOÁN", top_k=3)[0]
    assert hit.code == "E11.9" and hit.source == "birgermoell/icd10-clinical-notes"
    split1=make_pilot_examples(rows(), [r.code for r in recs])
    split2=make_pilot_examples(rows(), [r.code for r in recs])
    assert split1 == split2
    ids={ex["id"] for part in split1.values() for ex in part}
    assert ids == {"a","b"}
    for split, examples in split1.items():
        for ex in examples:
            assert ex["metadata"]["official_evaluation"] is False
            assert ex["positive_code"] not in ex["hard_negative_codes"]
