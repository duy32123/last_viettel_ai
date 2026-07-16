from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.assertion.metrics import multilabel_metrics
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("gold"); p.add_argument("pred")
    ns=p.parse_args(); gold=[json.loads(l) for l in Path(ns.gold).read_text(encoding="utf-8").splitlines() if l.strip()]; pred=[json.loads(l) for l in Path(ns.pred).read_text(encoding="utf-8").splitlines() if l.strip()]
    g=[e.get("assertions",[]) for r in gold for e in r.get("entities",[])]; pr=[e.get("assertions",[]) for r in pred for e in r.get("entities",[])]
    print(json.dumps(multilabel_metrics(g,pr), ensure_ascii=False, indent=2))
