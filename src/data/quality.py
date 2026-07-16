from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Any
from src.data.synthetic.generator import ann_hash
from src.data.dataset_schema import VALID_TYPES

TARGET_TYPES=set(VALID_TYPES)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p=Path(path)
    if not p.exists() or p.stat().st_size == 0: return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_quality_report(paths: Iterable[str | Path]) -> dict[str, Any]:
    rows=[]
    for p in paths: rows.extend(read_jsonl(p))
    docs=Counter(); ents=Counter(); assertions=Counter(); unmapped_ignore=Counter(); cand=Counter(); verified=Counter(); fams=defaultdict(set); dup=0; seen=set()
    train_types=Counter(); dev_types=Counter()
    for r in rows:
        split=r.get("source_split", "") or "unknown"; docs[(r.get("source","unknown"), split)] += 1
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
    }


def assert_data_gate(report: dict[str, Any], min_docs: int=1) -> None:
    docs=sum(report["documents_by_source_split"].values())
    if docs < min_docs: raise ValueError(f"not enough documents: {docs} < {min_docs}")
    if report["candidate_coverage"].get("fake_unverified_code", 0): raise ValueError("forbidden fake candidate code UNVERIFIED found")
    missing_train=TARGET_TYPES-set(report.get("train_entity_types", {}))
    missing_dev=TARGET_TYPES-set(report.get("dev_entity_types", {}))
    if missing_train or missing_dev:
        raise ValueError(f"missing train/dev samples for entity types: train={sorted(missing_train)} dev={sorted(missing_dev)}")
