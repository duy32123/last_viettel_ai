from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.weak_label import DEFAULT_QWEN_MODEL, TransformersQwenBackend, build_weak_corpus
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--train-real', default='data/processed/train.real.jsonl'); p.add_argument('--synthetic-train', default='data/processed/train.jsonl'); p.add_argument('--out-dir', default='data/processed'); p.add_argument('--annotation-dir', default='data/annotation'); p.add_argument('--min-confidence', type=float, default=0.9); p.add_argument('--qwen', action='store_true')
    p.add_argument('--qwen-model', default=DEFAULT_QWEN_MODEL); p.add_argument('--qwen-device', default='auto'); p.add_argument('--qwen-batch-size', type=int, default=4); p.add_argument('--qwen-cache', default='data/cache/qwen_weak_labels.json'); p.add_argument('--qwen-load-in-4bit', action='store_true'); p.add_argument('--qwen-max-candidates', type=int, default=0)
    ns=p.parse_args(); backend=None
    if ns.qwen:
        backend=TransformersQwenBackend(ns.qwen_model, ns.qwen_device, ns.qwen_load_in_4bit)
    print(json.dumps(build_weak_corpus(Path(ns.train_real), Path(ns.synthetic_train), Path(ns.out_dir), Path(ns.annotation_dir), ns.min_confidence, ns.qwen, backend, Path(ns.qwen_cache) if ns.qwen else None, ns.qwen_max_candidates, ns.qwen_batch_size), ensure_ascii=False, indent=2))
