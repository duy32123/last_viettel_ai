from __future__ import annotations
import hashlib, json, shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from src.data.adapters.span_jsonl import load_span_jsonl, DEFAULT_MAPPINGS
from src.data.synthetic.generator import ann_hash
from src.data.dataset_schema import validate_record, VALID_TYPES


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(65536), b''): h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False)+'\n')


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0: return []
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def dedupe(rows: list[dict]) -> tuple[list[dict], int]:
    seen=set(); out=[]; dup=0
    for r in rows:
        h=ann_hash(r)
        if h in seen: dup+=1; continue
        seen.add(h); out.append(r)
    return out, dup


def assert_no_upstream_leakage(splits: dict[str, list[dict]]) -> None:
    seen={}
    for split, rows in splits.items():
        for r in rows:
            uid=(r.get('source'), r.get('metadata',{}).get('upstream_id'))
            if uid[1] is None: continue
            if uid in seen and seen[uid] != split:
                raise ValueError(f"upstream document leakage: {uid} in {seen[uid]} and {split}")
            seen[uid]=split


def build_review_queue(real_rows: list[dict], limit: int) -> list[dict]:
    queue=[]
    for r in real_rows[:limit]:
        q=dict(r); q['metadata']=dict(q.get('metadata',{})); q['metadata']['gold_evaluation']=False
        q['review_status']='pending'; q['reviewer']=''; q['review_notes']=''; q['proposed_entities']=q.pop('entities', [])
        queue.append(q)
    return queue


def build_corpus(config_path: Path) -> dict[str, Any]:
    cfg=json.loads(config_path.read_text(encoding='utf-8'))
    out_dir=Path(cfg.get('output_dir','data/processed')); ann_dir=Path(cfg.get('annotation_dir','data/annotation'))
    out_dir.mkdir(parents=True, exist_ok=True); ann_dir.mkdir(parents=True, exist_ok=True)
    manifest=[]; reports=[]; real_by_split=defaultdict(list)
    for ds in cfg.get('datasets', []):
        local=Path(ds.get('local_path',''))
        terms=ds.get('license_terms','unspecified')
        entry={k:ds.get(k) for k in ['name','source_url','upstream_commit','license_terms']}
        if not ds.get('redistribution_allowed', False): entry['redistribution']='not_committed_raw_or_processed'
        if not local.exists():
            entry['status']='skipped_missing_local_data'; manifest.append(entry); continue
        entry['checksum_sha256']=sha256(local)
        mapping=ds.get('mapping') or DEFAULT_MAPPINGS.get(ds.get('adapter',''), {})
        if ds.get('adapter') in {'vimq','vietmed_ner'}:
            rows, report=load_span_jsonl(local, ds['name'], ds.get('split','train'), terms, mapping, synthetic=False)
        else:
            entry['status']='skipped_unsupported_adapter'; manifest.append(entry); continue
        real_by_split[ds.get('split','train')].extend(rows); reports.append({'name':ds['name'], **report}); entry['status']='loaded_local'; entry['records']=len(rows); manifest.append(entry)
    for split in list(real_by_split):
        real_by_split[split], _ = dedupe(real_by_split[split])
    assert_no_upstream_leakage(real_by_split)
    train_real=real_by_split.get('train', [])
    dev_prov=real_by_split.get('dev', []) + real_by_split.get('validation', [])
    synthetic=read_jsonl(Path(cfg.get('synthetic_train_path','data/processed/train.jsonl')))
    for r in synthetic:
        r.setdefault('metadata', {})['synthetic']=True
        r['metadata'].setdefault('upstream_id', r['id'])
    combined, combined_dups=dedupe(train_real + synthetic)
    pipeline_test=read_jsonl(Path(cfg.get('pipeline_test_source','data/processed/test.jsonl')))
    for r in pipeline_test:
        r.setdefault('metadata', {})['synthetic']=True
        r['metadata']['gold_evaluation']=False
    gold_todo=build_review_queue(dev_prov or train_real, int(cfg.get('gold_dev_sample_size', 100)))
    write_jsonl(out_dir/'train.real.jsonl', train_real)
    write_jsonl(out_dir/'train.combined.jsonl', combined)
    write_jsonl(out_dir/'dev.provisional.jsonl', dev_prov)
    write_jsonl(ann_dir/'gold_dev.todo.jsonl', gold_todo)
    write_jsonl(out_dir/'pipeline_test.jsonl', pipeline_test)
    report=corpus_report({'train.real':train_real,'dev.provisional':dev_prov,'train.combined':combined,'pipeline_test':pipeline_test}, reports, combined_dups, gold_todo)
    (out_dir/'real_corpus_manifest.json').write_text(json.dumps({'sources':manifest,'label_reports':reports,'report':report}, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def corpus_report(splits: dict[str,list[dict]], label_reports: list[dict], dups:int, gold_todo:list[dict]) -> dict[str,Any]:
    counts=Counter(); ents=Counter(); mapping=Counter(); real_seen=set(); synthetic_seen=set(); candidates=Counter()
    for split, rows in splits.items():
        counts[split]=len(rows)
        for r in rows:
            uid=(r.get('source'), r.get('metadata',{}).get('upstream_id', r.get('id')))
            if r.get('metadata',{}).get('synthetic'): synthetic_seen.add(uid)
            else: real_seen.add(uid)
            for e in r.get('entities',[]):
                ents[e['type']]+=1
                label=e.get('metadata',{}).get('source_label')
                if e['type']=='UNMAPPED': mapping['unmapped']+=1
                elif e['type'] in VALID_TYPES: mapping['direct_mapped']+=1
                if e.get('metadata',{}).get('needs_review'): mapping['review_needed']+=1
                candidates['with_candidate' if e.get('candidates') else 'without_candidate']+=1
    approved=sum(1 for r in gold_todo if r.get('review_status')=='approved')
    pending=sum(1 for r in gold_todo if r.get('review_status')=='pending')
    blockers=[]
    real=len(real_seen); synthetic=len(synthetic_seen)
    if real == 0: blockers.append('no real local dataset imported')
    if approved == 0: blockers.append('no human-approved gold dev records')
    return {'record_counts':dict(counts),'entity_counts':dict(ents),'mapping_counts':dict(mapping),'real_records':real,'synthetic_records':synthetic,'duplicate_records_removed':dups,'candidate_coverage':dict(candidates),'gold_dev':{'approved':approved,'pending':pending},'full_gate_blockers':blockers,'label_reports':label_reports}
