import json, random
from pathlib import Path
from src.data.import_hf_icd_aux import import_auxiliary_rows, make_pilot_examples, valid_icd10_code, assert_no_query_kb_leakage, evaluate_pilot_bm25
from src.linking.retrieval import LexicalIndex

def rows():
    return [
        {"id":"a","code":"E11.9","language":"vi","diagnosis":"Đái tháo đường type 2","journal_note":"Bệnh nhân đái tháo đường type 2 đang theo dõi."},
        {"id":"a-en","code":"E11.9","language":"en","diagnosis":"Type 2 diabetes mellitus","journal_note":"Patient followed for diabetes."},
        {"id":"b","code":"I10","language":"vi","diagnosis":"Tăng huyết áp","journal_note":"Ghi nhận tăng huyết áp."},
        {"id":"b","code":"I10","language":"vi","diagnosis":"Tăng huyết áp","journal_note":"Ghi nhận tăng huyết áp."},
        {"id":"c","code":"BAD","language":"en","diagnosis":"bad","journal_note":"bad"},
    ]

def test_icd10_format_auxiliary_schema_not_verified_and_report():
    assert valid_icd10_code("E11.9") and valid_icd10_code("I10") and not valid_icd10_code("BAD")
    recs, report=import_auxiliary_rows(rows(), version="pilot")
    assert report["source_kind"] == "auxiliary_hf" and report["official_kb"] is False and report["synthetic_notes"] is True
    assert report["license"] == "CC-BY-4.0" and report["verified"] == 0 and report["full_icd10_coverage"] is False
    assert report["records"] == 5 and report["unique_codes"] == 2 and report["vietnamese_records"] == 3
    assert report["unique_code_ratio"] == 2/5 and report["dataset_unique_code_coverage"] == 1.0
    assert report["invalid_code_formats"] == 1 and report["duplicate_aliases"] >= 1 and report["repeated_multilingual_code_rows"] >= 2
    assert all(r.verified is False for r in recs)
    e11=next(r for r in recs if r.code == "E11.9")
    assert e11.canonical_name == "Type 2 diabetes mellitus"
    assert "Bệnh nhân đái tháo đường type 2 đang theo dõi." not in [e11.canonical_name, *e11.aliases]
    assert all(p["source_field"] == "name" for p in e11.metadata["alias_provenance"])
    assert any(p["raw_alias"] == "Đái tháo đường type 2" and p["language"] == "vi" for p in e11.metadata["alias_provenance"])

def test_deterministic_canonical_leakage_gate_bm25_and_split_disjoint():
    base=rows(); shuffled=list(base); random.Random(7).shuffle(shuffled)
    recs1, _=import_auxiliary_rows(base, version="pilot"); recs2, _=import_auxiliary_rows(shuffled, version="pilot")
    assert [(r.code,r.canonical_name,sorted(r.aliases)) for r in recs1] == [(r.code,r.canonical_name,sorted(r.aliases)) for r in recs2]
    names={r.code:r.canonical_name for r in recs1}
    split1=make_pilot_examples(base, [r.code for r in recs1], names)
    split2=make_pilot_examples(base, [r.code for r in recs1], names)
    assert split1 == split2
    assert assert_no_query_kb_leakage(recs1, split1)["query_kb_overlap"] == 0
    ids_by_split={s:{ex["id"] for ex in xs} for s,xs in split1.items()}
    assert ids_by_split["train"].isdisjoint(ids_by_split["dev"] | ids_by_split["test"])
    metrics=evaluate_pilot_bm25(recs1, split1, top_k=10)
    assert metrics["candidate_count"] == len(recs1) == 2 and metrics["official_evaluation"] is False
    assert set(metrics["overall"]) == {"recall@1","recall@5","recall@10","mrr"}
    assert "vi" in metrics["per_language"]
    for part in split1.values():
        for ex in part:
            assert ex["positive_code"] not in ex["hard_negative_codes"]

def test_auxiliary_alias_retrieval_and_config_no_path_required(tmp_path, monkeypatch):
    import sys
    import scripts.prepare_linking_kb as prep
    recs, _=import_auxiliary_rows(rows(), version="pilot")
    idx=LexicalIndex(recs, include_unverified=True)
    hit=idx.search("đái tháo đường type 2", "CHẨN_ĐOÁN", top_k=3)[0]
    assert hit.code == "E11.9" and hit.source == "birgermoell/icd10-clinical-notes"
    cfg=json.loads(Path('configs/linking_kb.hf_auxiliary.yaml').read_text(encoding='utf-8'))
    assert cfg['sources'][0]['enabled'] is True and 'path' not in cfg['sources'][0]
    cfg['output_dir']=str(tmp_path/'kb')
    cfg_path=tmp_path/'hf_aux.yaml'; cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding='utf-8')
    monkeypatch.setattr(prep, 'rows_from_hf', lambda split=None: rows())
    monkeypatch.setattr(sys, 'argv', ['prepare_linking_kb.py','--config',str(cfg_path)])
    prep.main()
    out=Path(cfg['output_dir'])
    assert (out/'icd10.jsonl').exists() and (out/'manifest.json').exists() and (out/'auxiliary_pilot_examples.json').exists()
    entries=[json.loads(l) for l in (out/'icd10.jsonl').read_text(encoding='utf-8').splitlines()]
    assert entries and all(e['verified'] is False and e['metadata']['official_kb'] is False for e in entries)
