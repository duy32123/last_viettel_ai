from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.assertion.inference import predict_assertions, load_thresholds
from src.models.assertion.preprocess import register_special_tokens
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("input"); p.add_argument("--output", required=True); p.add_argument("--model"); p.add_argument("--thresholds"); p.add_argument("--batch-size", type=int, default=16)
    ns=p.parse_args(); rows=[json.loads(l) for l in Path(ns.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    threshold_path=ns.thresholds or (str(Path(ns.model)/"thresholds.json") if ns.model else None)
    thresholds=load_thresholds(threshold_path)
    model=tokenizer=None
    if ns.model:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tokenizer=AutoTokenizer.from_pretrained(ns.model); model=AutoModelForSequenceClassification.from_pretrained(ns.model)
            register_special_tokens(tokenizer, model)
        except Exception as e:
            raise RuntimeError("Install requirements-train.txt to run model-backed assertion inference") from e
    out=[]
    for r in rows:
        entities=predict_assertions(r["text"], r.get("entities", []), model=model, tokenizer=tokenizer, thresholds=thresholds, batch_size=ns.batch_size)
        out.append({**r,"entities":entities})
    Path(ns.output).write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in out)+"\n", encoding="utf-8")
