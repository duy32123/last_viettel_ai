from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.assertion_generator import AssertionGenConfig, build_assertion_corpus
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config", default="configs/assertion_data.yaml"); p.add_argument("--out-dir", default="data/processed/assertion"); p.add_argument("--audit", default="data/annotation/assertion_audit.todo.jsonl")
    ns=p.parse_args(); report=build_assertion_corpus(ns.out_dir, ns.audit, AssertionGenConfig())
    print(json.dumps(report, ensure_ascii=False, indent=2))
