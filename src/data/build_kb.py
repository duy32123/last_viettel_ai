from __future__ import annotations

import argparse, json
from datetime import date
from pathlib import Path

from .import_icd import import_icd_csv
from .import_rxnorm import import_rxnorm_csv
from .kb_schema import dedupe_records, file_sha256, write_jsonl


def build(config_path: Path) -> dict[str, int]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    out_dir = Path(cfg.get("output_dir", "data/processed/kb")); out_dir.mkdir(parents=True, exist_ok=True)
    manifests=[]; counts={}
    by_term={"icd": [], "rxnorm": []}
    for src in cfg.get("sources", []):
        path=Path(src.get("path", ""))
        if not path.exists():
            manifests.append({**src, "status":"skipped_missing", "fetched_at": str(date.today())}); continue
        if src["kind"] == "icd_csv":
            recs=import_icd_csv(path, src["terminology"], str(src["version"]), src["name"], bool(src.get("verified", True))); by_term["icd"].extend(recs)
        elif src["kind"] == "rxnorm_csv":
            recs=import_rxnorm_csv(path, str(src["version"]), src["name"], bool(src.get("verified", True))); by_term["rxnorm"].extend(recs)
        else:
            manifests.append({**src, "status":"skipped_unsupported", "fetched_at": str(date.today())}); continue
        manifests.append({**src, "status":"loaded", "checksum_sha256": file_sha256(path), "fetched_at": str(date.today())})
    for name, recs in by_term.items():
        deduped=dedupe_records(recs)
        if deduped: write_jsonl(out_dir / f"{name}.jsonl", deduped)
        counts[name]=len(deduped)
    (out_dir / "manifest.json").write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")
    return counts


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--config", default="configs/data_sources.yaml")
    print(json.dumps(build(Path(p.parse_args().config)), ensure_ascii=False))
if __name__ == "__main__": main()
