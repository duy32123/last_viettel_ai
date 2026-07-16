from __future__ import annotations
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any
from src.data.synthetic.generator import ann_hash
from src.data.dataset_schema import VALID_TYPES

TARGET_TYPES=set(VALID_TYPES)

@dataclass
class DataGateConfig:
    mode: str = "smoke"
    min_train_documents: int = 1
    min_dev_documents: int = 1
    min_entities_per_type_train: int = 1
    min_entities_per_type_dev: int = 1
    require_non_synthetic_source: bool = False
    require_gold_dev: bool = False
    forbid_synthetic_test_as_official_eval: bool = True


def load_gate_config(path: str | Path | None=None, overrides: dict[str, Any] | None=None) -> DataGateConfig:
    data={}
    if path:
        data=json.loads(Path(path).read_text(encoding="utf-8"))
    if overrides: data.update(overrides)
    return DataGateConfig(**{k:v for k,v in data.items() if k in DataGateConfig.__annotations__})


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p=Path(path)
    if not p.exists() or p.stat().st_size == 0: return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_quality_report(paths: Iterable[str | Path]) -> dict[str, Any]:
    rows=[]
    for p in paths: rows.extend(read_jsonl(p))
    docs=Counter(); ents=Counter(); assertions=Counter(); unmapped_ignore=Counter(); cand=Counter(); verified=Counter(); fams=defaultdict(set); dup=0; seen=set()
    train_types=Counter(); dev_types=Counter(); split_docs=Counter(); sources=Counter(); gold_dev=Counter(); synthetic_test=0
    for r in rows:
        split=r.get("source_split", "") or "unknown"; source=r.get("source","unknown")
        docs[(source, split)] += 1; split_docs[split] += 1; sources[source] += 1
        if split in {"dev", "validation"}:
            gold_dev["gold" if r.get("metadata", {}).get("gold_evaluation") is True else "not_gold"] += 1
        if split == "test" and source.startswith("synthetic") and r.get("metadata", {}).get("gold_evaluation") is not True:
            synthetic_test += 1
        h=ann_hash(r)
        if h in seen: dup += 1
        seen.add(h)
        if r.get("metadata", {}).get("template_family"): fams[split].add(r["metadata"]["template_family"])
        for e in r.get("entities", []):
            typ=e["type"]; ents[typ]+=1
            if split == "train" and typ in TARGET_TYPES: train_types[typ]+=1
            if split in {"dev", "validation"} and typ in TARGET_TYPES: dev_types[typ]+=1
            if typ in {"UNMAPPED","IGNORE"}: unmapped_ignore[typ]+=1
            for a in e.get("assertions",[]): assertions[a]+=1
            cs=e.get("candidates", [])
            if cs: cand["with_candidate"] += 1
            else: cand["without_candidate"] += 1
            for c in cs:
                if isinstance(c, str):
                    if c == "UNVERIFIED": cand["fake_unverified_code"] += 1
                    verified["unknown_string"] += 1
                else:
                    if c.get("code") == "UNVERIFIED": cand["fake_unverified_code"] += 1
                    verified["verified" if c.get("verified") else "unverified"] += 1
    total=len(rows)
    return {
        "documents_by_source_split": {f"{k[0]}::{k[1]}": v for k,v in sorted(docs.items())},
        "documents_by_split": dict(sorted(split_docs.items())),
        "sources": dict(sorted(sources.items())),
        "entities_by_type": dict(sorted(ents.items())),
        "assertion_distribution": dict(sorted(assertions.items())),
        "duplicate_records": dup,
        "duplicate_rate": dup / total if total else 0.0,
        "unmapped_ignore_counts": dict(sorted(unmapped_ignore.items())),
        "unmapped_ignore_rate": sum(unmapped_ignore.values()) / sum(ents.values()) if ents else 0.0,
        "candidate_coverage": dict(cand),
        "candidate_coverage_rate": cand["with_candidate"] / (cand["with_candidate"] + cand["without_candidate"]) if (cand["with_candidate"] + cand["without_candidate"]) else 0.0,
        "verified_candidate_counts": dict(verified),
        "template_families_by_split": {k: len(v) for k,v in sorted(fams.items())},
        "train_entity_types": dict(train_types),
        "dev_entity_types": dict(dev_types),
        "dev_gold_evaluation": dict(gold_dev),
        "synthetic_test_not_gold_records": synthetic_test,
    }


def assert_data_gate(report: dict[str, Any], config: DataGateConfig | dict[str, Any] | None=None, min_docs: int | None=None) -> None:
    if config is None:
        cfg=DataGateConfig(min_train_documents=min_docs or 1, min_dev_documents=0 if min_docs else 1)
    elif isinstance(config, dict):
        cfg=DataGateConfig(**config)
    else:
        cfg=config
    errors=[]
    split_docs=report.get("documents_by_split", {})
    if split_docs.get("train", 0) < cfg.min_train_documents:
        errors.append(f"not enough train documents: {split_docs.get('train',0)} < {cfg.min_train_documents}")
    dev_count=split_docs.get("dev", 0) + split_docs.get("validation", 0)
    if dev_count < cfg.min_dev_documents:
        errors.append(f"not enough dev documents: {dev_count} < {cfg.min_dev_documents}")
    if report["candidate_coverage"].get("fake_unverified_code", 0):
        errors.append("forbidden fake candidate code UNVERIFIED found")
    train_counts=report.get("train_entity_types", {}); dev_counts=report.get("dev_entity_types", {})
    missing_train=[t for t in sorted(TARGET_TYPES) if train_counts.get(t,0) < cfg.min_entities_per_type_train]
    missing_dev=[t for t in sorted(TARGET_TYPES) if dev_counts.get(t,0) < cfg.min_entities_per_type_dev]
    if missing_train or missing_dev:
        errors.append(f"missing train/dev entity minima: train={missing_train} dev={missing_dev}")
    if cfg.require_non_synthetic_source and not any(not s.startswith("synthetic") for s in report.get("sources", {})):
        errors.append("full-training gate requires at least one non-synthetic source")
    if cfg.require_gold_dev and report.get("dev_gold_evaluation", {}).get("not_gold", 0):
        errors.append("full-training gate requires all dev records to have metadata.gold_evaluation=true")
    if cfg.forbid_synthetic_test_as_official_eval and cfg.mode == "full" and report.get("synthetic_test_not_gold_records", 0):
        errors.append("synthetic test records are pipeline checks only, not official evaluation")
    if errors:
        raise ValueError("; ".join(errors))
