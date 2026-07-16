from pathlib import Path
import argparse, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dicts import DIAGNOSIS_DICT, DRUG_DICT
from src.data.kb_schema import KBRecord, write_jsonl

OUT = Path("data/processed/kb")

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--icd-terminology", default="ICD-10", choices=["ICD-10","ICD-10-CM"])
    p.add_argument("--out-dir", default=str(OUT))
    ns=p.parse_args()
    out=Path(ns.out_dir)
    meta={"legacy_warning":"best-effort unverified baseline candidate","variant_unknown":True}
    icd=[]
    for name, codes in DIAGNOSIS_DICT.items():
        for code in codes:
            icd.append(KBRecord(code, name, [], ns.icd_terminology, "legacy-unverified", "legacy_dicts", False, dict(meta)))
    rx=[KBRecord(code, name, [], "RxNorm", "legacy-unverified", "legacy_dicts", False, {"legacy_warning":"best-effort unverified baseline candidate"}) for name, code in DRUG_DICT.items()]
    write_jsonl(out/"icd10_seed.jsonl", icd); write_jsonl(out/"rxnorm_seed.jsonl", rx)
    print(f"exported {len(icd)} ICD seed rows as {ns.icd_terminology} and {len(rx)} RxNorm seed rows")
if __name__ == "__main__": main()
