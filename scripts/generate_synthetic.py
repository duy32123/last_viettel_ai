from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.synthetic.generator import generate

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config", default="configs/synthetic.yaml"); ns=p.parse_args()
    cfg=json.loads(Path(ns.config).read_text(encoding="utf-8"))
    print(json.dumps(generate(int(cfg.get("seed",13)), cfg.get("counts",{}), Path(cfg.get("output_dir","data/processed")), cfg.get("kb_paths")), ensure_ascii=False))
if __name__ == "__main__": main()
