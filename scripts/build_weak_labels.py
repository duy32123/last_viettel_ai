from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.weak_label import build_weak_corpus
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--train-real', default='data/processed/train.real.jsonl'); p.add_argument('--synthetic-train', default='data/processed/train.jsonl'); p.add_argument('--out-dir', default='data/processed'); p.add_argument('--annotation-dir', default='data/annotation'); p.add_argument('--min-confidence', type=float, default=0.9); p.add_argument('--qwen', action='store_true')
    ns=p.parse_args(); print(json.dumps(build_weak_corpus(Path(ns.train_real), Path(ns.synthetic_train), Path(ns.out_dir), Path(ns.annotation_dir), ns.min_confidence, ns.qwen), ensure_ascii=False, indent=2))
