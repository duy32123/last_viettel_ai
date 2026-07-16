from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.corpus import build_corpus
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config', default='configs/real_corpus.yaml')
    ns=p.parse_args(); print(json.dumps(build_corpus(Path(ns.config)), ensure_ascii=False, indent=2))
