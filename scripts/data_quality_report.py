from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.quality import build_quality_report, assert_data_gate

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("paths", nargs="+"); p.add_argument("--gate", action="store_true"); p.add_argument("--out")
    ns=p.parse_args(); report=build_quality_report(ns.paths)
    if ns.gate: assert_data_gate(report)
    text=json.dumps(report, ensure_ascii=False, indent=2)
    if ns.out: Path(ns.out).write_text(text, encoding="utf-8")
    print(text)
