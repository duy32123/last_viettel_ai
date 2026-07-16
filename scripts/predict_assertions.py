from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.assertion.inference import predict_assertions
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("input"); p.add_argument("--output", required=True)
    ns=p.parse_args(); rows=[json.loads(l) for l in Path(ns.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    out=[]
    for r in rows: out.append({**r,"entities":predict_assertions(r["text"], r.get("entities", []))})
    Path(ns.output).write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in out)+"\n", encoding="utf-8")
