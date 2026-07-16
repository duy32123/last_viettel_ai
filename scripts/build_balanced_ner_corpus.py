from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.balanced_ner_corpus import BalanceConfig, build_balanced_corpus

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train-silver", default="data/processed/train.silver.jsonl")
    p.add_argument("--train-synthetic", default="data/processed/train.jsonl")
    p.add_argument("--output", default="data/processed/train.v2.balanced.jsonl")
    p.add_argument("--audit-output", default="data/annotation/audit_v2.todo.jsonl")
    p.add_argument("--report", default="data/processed/balanced_v2_report.json")
    p.add_argument("--min-entities-per-type", type=int, default=500)
    p.add_argument("--min-canonical-concepts-per-type", type=int, default=15)
    p.add_argument("--min-canonical-concepts-drug-diagnosis", type=int, default=20)
    p.add_argument("--max-class-imbalance-ratio", type=float, default=3.0)
    p.add_argument("--max-top-canonical-share", type=float, default=0.07)
    p.add_argument("--audit-samples-per-type", type=int, default=50)
    p.add_argument("--seed", type=int, default=57)
    p.add_argument("--profile", choices=["experimental", "full"], default="experimental")
    ns=p.parse_args()
    cfg=BalanceConfig(ns.min_entities_per_type, ns.min_canonical_concepts_per_type, ns.min_canonical_concepts_drug_diagnosis, ns.max_class_imbalance_ratio, ns.max_top_canonical_share, ns.audit_samples_per_type, ns.seed, ns.profile)
    report=build_balanced_corpus(Path(ns.train_silver), Path(ns.train_synthetic), Path(ns.output), Path(ns.audit_output), Path(ns.report), cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2))
