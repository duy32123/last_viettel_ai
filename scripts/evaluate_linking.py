from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.linking.pipeline import load_jsonl
from src.linking.metrics import evaluate
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--gold', required=True); p.add_argument('--pred', required=True); ns=p.parse_args(); print(json.dumps(evaluate(load_jsonl(ns.gold), load_jsonl(ns.pred)), ensure_ascii=False, indent=2))
