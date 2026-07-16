from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dicts import DIAGNOSIS_DICT, DRUG_DICT
from src.data.kb_schema import KBRecord, write_jsonl

OUT = Path("data/processed/kb")

def main() -> None:
    icd=[]
    for name, codes in DIAGNOSIS_DICT.items():
        for code in codes:
            icd.append(KBRecord(code, name, [], "ICD-10-CM" if "." in code or code.endswith("A") else "ICD-10", "legacy-unverified", "legacy_dicts", False, {"legacy_warning":"best-effort unverified baseline candidate"}))
    rx=[KBRecord(code, name, [], "RxNorm", "legacy-unverified", "legacy_dicts", False, {"legacy_warning":"best-effort unverified baseline candidate"}) for name, code in DRUG_DICT.items()]
    write_jsonl(OUT/"icd10_seed.jsonl", icd); write_jsonl(OUT/"rxnorm_seed.jsonl", rx)
    print(f"exported {len(icd)} ICD seed rows and {len(rx)} RxNorm seed rows")
if __name__ == "__main__": main()
