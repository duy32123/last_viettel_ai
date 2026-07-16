from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.gold_dev import promote
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--todo', default='data/annotation/gold_dev.todo.jsonl'); p.add_argument('--out', default='data/processed/dev.gold.jsonl')
    ns=p.parse_args(); print(json.dumps(promote(Path(ns.todo), Path(ns.out)), ensure_ascii=False))
