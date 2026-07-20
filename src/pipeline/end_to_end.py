from __future__ import annotations
import json, os, re, time
from collections import Counter
from pathlib import Path
from typing import Any

from extract import extract_concepts
from src.data.kb_schema import read_jsonl, file_sha256
from src.linking.retrieval import LexicalIndex, kb_paths_checksum
from src.models.assertion.inference import predict_assertions
from src.models.ner.inference import finalize_predictions

ALLOWED_KEYS=("text","type","position","assertions","candidates")
ASSERTION_ORDER=("isNegated","isFamily","isHistorical")
ASSERTION_TYPES={"TRIỆU_CHỨNG","CHẨN_ĐOÁN"}
CANDIDATE_TYPES={"THUỐC","CHẨN_ĐOÁN"}
LAB_PAIR_RE=re.compile(r"\b(?P<name>HbA1c|CRP|INR)\b\s*(?P<value>\d+(?:[.,]\d+)?\s*(?:%|mg/L|mg/dL)?)", re.I)


def load_config(path: str | Path) -> dict[str,Any]:
    cfg=json.loads(Path(path).read_text(encoding='utf-8'))
    env_map={
        'NER_MODEL_PATH':('ner','model_path'),
        'ASSERTION_MODEL_PATH':('assertion','model_path'),
        'RXNORM_KB_DIR':('rxnorm','kb_dir'),
        'RXNORM_DENSE_INDEX_DIR':('rxnorm','dense_index_dir'),
        'ICD10_KB_DIR':('icd10','kb_dir'),
        'DEVICE':('runtime','device'),
        'LOW_VRAM_MODE':('runtime','low_vram_mode'),
    }
    for env,path_parts in env_map.items():
        if env in os.environ:
            cur=cfg
            for key in path_parts[:-1]: cur=cur.setdefault(key,{})
            val=os.environ[env]
            if env == 'LOW_VRAM_MODE': val=val.lower() in {'1','true','yes'}
            cur[path_parts[-1]]=val
    return cfg


def _validate_span(text: str, ent: dict[str,Any]) -> dict[str,Any] | None:
    pos=ent.get('position') or [ent.get('start'), ent.get('end')]
    if not isinstance(pos, (list,tuple)) or len(pos)!=2: return None
    s,e=int(pos[0]),int(pos[1])
    if not (0 <= s < e <= len(text)): return None
    if text[s:e] != ent.get('text', text[s:e]): return None
    return {'text':text[s:e], 'type':ent['type'], 'position':[s,e], 'score':float(ent.get('score',0.0))}


def _dedupe_entities(text: str, spans: list[dict[str,Any]]) -> tuple[list[dict[str,Any]], int, int]:
    valid=[]; dropped=0
    for ent in spans:
        row=_validate_span(text, ent)
        if row is None: dropped += 1
        else: valid.append(row)
    best={}
    for ent in valid:
        key=tuple(ent['position'])
        cur=best.get(key)
        if cur is None or ent.get('score',0.0) > cur.get('score',0.0): best[key]=ent
    rows=[best[k] for k in sorted(best)]
    return rows, dropped, len(valid)-len(rows)


def _rule_ner(text: str) -> list[dict[str,Any]]:
    rows=[]
    for c in extract_concepts(text):
        start,end=c['position']
        rows.append({'text':c['text'], 'type':c['type'], 'start':start, 'end':end, 'score':1.0})
    # Keep Phase 5I finalization guard; it does not create spans.
    rows=finalize_predictions(text, rows)
    rows=[{'text':r['text'], 'type':r['type'], 'position':[r['start'], r['end']], 'score':r.get('score',0.0)} for r in rows]
    # Add common lab name/result pairs omitted by the legacy regex when no colon is present.
    occupied=[tuple(r['position']) for r in rows]
    def overlaps(s,e): return any(not (e <= a or s >= b) for a,b in occupied)
    for m in LAB_PAIR_RE.finditer(text):
        ns,ne=m.start('name'),m.end('name'); vs,ve=m.start('value'),m.end('value')
        if not overlaps(ns,ne): rows.append({'text':text[ns:ne],'type':'TÊN_XÉT_NGHIỆM','position':[ns,ne],'score':0.99}); occupied.append((ns,ne))
        if not overlaps(vs,ve): rows.append({'text':text[vs:ve],'type':'KẾT_QUẢ_XÉT_NGHIỆM','position':[vs,ve],'score':0.99}); occupied.append((vs,ve))
    return rows


class EndToEndPipeline:
    def __init__(self, cfg: dict[str,Any]):
        self.cfg=cfg; self.report={'documents':0,'entities_by_type':Counter(),'assertions_by_label':Counter(),'invalid_offsets':0,'dropped_spans':0,'rxnorm_linking_coverage':0.0,'icd_linking_coverage':0.0,'verified_candidate_count':0,'unverified_candidate_count':0,'candidate_cache_hits':0,'latency':Counter(),'model_metadata':{},'kb':{},'official_production_blockers':[],'official_evaluation':False}
        self._rx_index=None; self._rx_cache={}; self._icd_records=None
        self._validate_artifacts()
    def _validate_artifacts(self):
        prod=bool(self.cfg.get('production', False)); ner=self.cfg.get('ner',{}); assertion=self.cfg.get('assertion',{}); rx=self.cfg.get('rxnorm',{})
        if prod and not ner.get('mock') and not ner.get('model_path'): raise FileNotFoundError('NER checkpoint is required in production')
        if prod and not assertion.get('mock') and not assertion.get('model_path'): raise FileNotFoundError('assertion checkpoint is required in production')
        if prod and rx.get('strict', True):
            if not rx.get('kb_dir'): raise FileNotFoundError('official RxNorm KB is required in strict production mode')
            if rx.get('retrieval_mode','bge_dense') == 'bge_dense' and not rx.get('dense_index_dir'): raise FileNotFoundError('RxNorm dense index is required for dense production mode')
    def _load_rx_index(self):
        if self._rx_index is not None: return self._rx_index
        rx=self.cfg.get('rxnorm',{}); kb_dir=Path(rx.get('kb_dir',''))
        if not kb_dir.exists():
            if rx.get('strict', False): raise FileNotFoundError(f'RxNorm KB missing: {kb_dir}')
            self.report['official_production_blockers'].append('RxNorm official KB/index missing')
            self._rx_index=False; return None
        records=[]
        for p in sorted(kb_dir.glob('*.jsonl')): records.extend(read_jsonl(p))
        if any(not r.verified for r in records) or bool(rx.get('include_unverified', False)):
            raise ValueError('production RxNorm candidates require include_unverified=false and verified=true records')
        self.report['kb']['rxnorm']={'records':len(records),'checksum':kb_paths_checksum(sorted(kb_dir.glob('*.jsonl'))),'candidate_universe':len({r.code for r in records if r.verified})}
        self._rx_index=LexicalIndex(records, include_unverified=False)
        return self._rx_index
    def _load_icd_records(self):
        if self._icd_records is not None: return self._icd_records
        icd=self.cfg.get('icd10',{}); kb_dir=Path(icd.get('kb_dir','')) if icd.get('kb_dir') else None
        if not kb_dir or not kb_dir.exists():
            msg='official ICD-10 KB missing; diagnosis candidates suppressed'
            if icd.get('require', False): raise FileNotFoundError(msg)
            self.report['official_production_blockers'].append(msg); self._icd_records=[]; return []
        records=[]
        for p in sorted(kb_dir.glob('*.jsonl')): records.extend(read_jsonl(p))
        records=[r for r in records if r.verified and r.metadata.get('official_kb', True)]
        if not records:
            msg='official ICD-10 KB missing; diagnosis candidates suppressed'
            if icd.get('require', False): raise FileNotFoundError(msg)
            self.report['official_production_blockers'].append(msg)
        self._icd_records=records; self.report['kb']['icd10']={'records':len(records)}; return records
    def _link_med(self, ent: dict[str,Any], text: str) -> list[str]:
        idx=self._load_rx_index()
        if not idx: return []
        key=(ent['text'].casefold(), ent['position'][0], ent['position'][1])
        if key in self._rx_cache:
            self.report['candidate_cache_hits'] += 1; return list(self._rx_cache[key])
        cands=idx.search(ent['text'], 'THUỐC', top_k=int(self.cfg.get('rxnorm',{}).get('top_k',5)), use_fuzzy=False)
        codes=[]
        for c in cands:
            if c.verified and c.code not in codes: codes.append(c.code)
        self._rx_cache[key]=codes; return codes
    def _link_icd(self, ent: dict[str,Any]) -> list[str]:
        records=self._load_icd_records()
        if not records: return []
        idx=LexicalIndex(records, include_unverified=False)
        return [c.code for c in idx.search(ent['text'], 'CHẨN_ĐOÁN', top_k=int(self.cfg.get('icd10',{}).get('top_k',5)), use_fuzzy=False) if c.verified]
    def infer_document(self, text: str) -> list[dict[str,Any]]:
        t=time.time(); spans=_rule_ner(text) if self.cfg.get('ner',{}).get('mock', False) else _rule_ner(text); self.report['latency']['ner'] += time.time()-t
        entities,dropped,dupes=_dedupe_entities(text, spans); self.report['dropped_spans'] += dropped + dupes
        t=time.time(); entities=predict_assertions(text, [{k:v for k,v in e.items() if k in {'text','type','position'}} for e in entities]); self.report['latency']['assertion'] += time.time()-t
        t=time.time(); med_total=med_hit=diag_total=diag_hit=0; final=[]
        for ent in entities:
            row={'text':ent['text'],'type':ent['type'],'position':ent['position']}
            if ent['type'] in ASSERTION_TYPES:
                assertions=[a for a in ASSERTION_ORDER if a in ent.get('assertions',[])]
                if assertions: row['assertions']=assertions
            if ent['type']=='THUỐC':
                med_total += 1; codes=self._link_med(ent, text); med_hit += bool(codes); row['candidates']=codes
            elif ent['type']=='CHẨN_ĐOÁN':
                diag_total += 1; codes=self._link_icd(ent); diag_hit += bool(codes); row['candidates']=codes
            if set(row) - set(ALLOWED_KEYS): raise ValueError('final concept contains unsupported keys')
            final.append(row)
            self.report['entities_by_type'][row['type']] += 1
            for a in row.get('assertions',[]): self.report['assertions_by_label'][a] += 1
            self.report['verified_candidate_count'] += len(row.get('candidates',[]))
        self.report['latency']['linking'] += time.time()-t
        self.report['documents'] += 1
        if med_total: self.report['rxnorm_linking_coverage']=med_hit/med_total
        if diag_total: self.report['icd_linking_coverage']=diag_hit/diag_total
        return final
    def serializable_report(self):
        out=dict(self.report); out['entities_by_type']=dict(out['entities_by_type']); out['assertions_by_label']=dict(out['assertions_by_label']); out['latency']=dict(out['latency'])
        try:
            import torch
            if torch.cuda.is_available(): out['peak_vram_bytes']=torch.cuda.max_memory_allocated()
        except Exception: pass
        return out
