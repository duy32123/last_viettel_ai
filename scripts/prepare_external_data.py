from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.adapters.simple_jsonl import load_jsonl
from src.data.adapters.phoner_covid19 import load_native as load_phoner_native

def main():
    a=argparse.ArgumentParser(); a.add_argument("--config", default="configs/datasets.yaml"); ns=a.parse_args()
    cfg=json.loads(Path(ns.config).read_text(encoding="utf-8")); out=Path(cfg.get("output","data/processed/external.jsonl")); out.parent.mkdir(parents=True, exist_ok=True)
    rows=[]; report=[]
    for d in cfg.get("datasets", []):
        raw_path=d.get("fixture_path") or d.get("path")
        if not raw_path:
            report.append({"name":d["name"],"status":"registered_only_skipped_no_local_data"}); continue
        p=Path(raw_path)
        if not p.exists(): report.append({"name":d["name"],"status":"skipped_missing_or_license"}); continue
        if d.get("adapter") == "phoner_covid19_native":
            loaded=load_phoner_native(p,d["name"],d.get("split","train"),d.get("license","unknown"),d.get("mapping",{}))
        else:
            loaded=load_jsonl(p,d["name"],d.get("split","train"),d.get("license","unknown"))
        rows += loaded; report.append({"name":d["name"],"status":"loaded","records":len(loaded)})
    with out.open("w", encoding="utf-8") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(json.dumps(report, ensure_ascii=False))
if __name__ == "__main__": main()
