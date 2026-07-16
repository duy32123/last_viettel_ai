from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.assertion.preprocess import make_examples

def load_jsonl(path):
    p=Path(path); return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config", default="configs/train_assertion.xlmr_base.yaml"); p.add_argument("--dry-run-smoke", action="store_true"); p.add_argument("--resume-from-checkpoint")
    ns=p.parse_args()
    cfg=json.loads(Path(ns.config).read_text(encoding="utf-8"))
    train=make_examples(load_jsonl(cfg["train_path"])); dev=make_examples(load_jsonl(cfg["dev_path"]))
    if ns.dry_run_smoke:
        print(json.dumps({"train_examples":len(train),"dev_examples":len(dev),"model_name":cfg["model_name"],"dry_run":True}, ensure_ascii=False)); sys.exit(0)
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    except Exception as e:
        raise RuntimeError("Install requirements-train.txt to train assertion model") from e
    raise RuntimeError("Full assertion fine-tune is not run in Codex; use the printed command in the PR report.")
