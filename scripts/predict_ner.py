from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.ner.inference import predict_with_model


def texts_from_path(path: Path):
    if path.is_dir():
        for p in sorted(path.glob("*.txt")): yield p.stem, p.read_text(encoding="utf-8")
    else:
        yield path.stem, path.read_text(encoding="utf-8")

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("input"); p.add_argument("--model", required=True); p.add_argument("--output", required=True); p.add_argument("--max-length", type=int, default=256); p.add_argument("--stride", type=int, default=64)
    ns=p.parse_args()
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
    except Exception as e:
        raise RuntimeError("transformers is required for NER prediction") from e
    tok=AutoTokenizer.from_pretrained(ns.model, use_fast=True); model=AutoModelForTokenClassification.from_pretrained(ns.model)
    out=Path(ns.output); out.mkdir(parents=True, exist_ok=True)
    for doc_id,text in texts_from_path(Path(ns.input)):
        ents=[]
        for e in predict_with_model(text, tok, model, model.config.id2label, ns.max_length, ns.stride):
            ent={"text": text[e["start"]:e["end"]], "type": e["type"], "position": [e["start"], e["end"]]}
            assert ent["text"] == text[ent["position"][0]:ent["position"][1]]
            ents.append(ent)
        (out/f"{doc_id}.json").write_text(json.dumps(ents, ensure_ascii=False, indent=2), encoding="utf-8")
