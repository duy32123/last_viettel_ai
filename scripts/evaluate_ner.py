from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.ner.metrics import evaluate_spans
from src.models.ner.preprocess import read_jsonl

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--gold", required=True); p.add_argument("--pred", required=True); p.add_argument("--errors-out", required=True)
    ns=p.parse_args(); result=evaluate_spans(read_jsonl(ns.gold), read_jsonl(ns.pred))
    Path(ns.errors_out).write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in result.pop("errors"))+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
