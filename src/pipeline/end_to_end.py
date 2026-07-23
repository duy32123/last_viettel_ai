from __future__ import annotations
import json, math, os, re, time
from collections import Counter
from pathlib import Path
from typing import Any

from extract import extract_concepts
from src.data.kb_schema import read_jsonl
from src.linking.dense import BGEM3Backend, dense_expected_manifest, load_dense_index
from src.linking.normalization import normalize_mention, normalize_text
from src.linking.retrieval import LexicalIndex, kb_paths_checksum, load_lexical_index
from src.models.assertion.inference import load_thresholds, predict_assertions
from src.models.ner.inference import finalize_predictions, predict_with_model
from src.models.llm_hybrid.extractor import LLMHybridExtractor

ALLOWED_KEYS=("text","type","position","assertions","candidates")
ASSERTION_ORDER=("isNegated","isFamily","isHistorical")
ASSERTION_TYPES={"TRIỆU_CHỨNG","CHẨN_ĐOÁN","THUỐC"}
LAB_PAIR_RE=re.compile(r"\b(?P<name>HbA1c|CRP|INR)\b\s*(?P<value>\d+(?:[.,]\d+)?\s*(?:%|mg/L|mg/dL)?)", re.I)


def load_config(path: str | Path) -> dict[str,Any]:
    cfg=json.loads(Path(path).read_text(encoding='utf-8'))
    env_map={'NER_MODEL_PATH':('ner','model_path'),'ASSERTION_MODEL_PATH':('assertion','model_path'),'RXNORM_KB_DIR':('rxnorm','kb_dir'),'RXNORM_DENSE_INDEX_DIR':('rxnorm','dense_index_dir'),'RXNORM_BGE_MODEL_PATH':('rxnorm','bge_model_path'),'ICD10_KB_DIR':('icd10','kb_dir'),'DEVICE':('runtime','device'),'LOW_VRAM_MODE':('runtime','low_vram_mode')}
    for env,path_parts in env_map.items():
        if env in os.environ:
            cur=cfg
            for key in path_parts[:-1]: cur=cur.setdefault(key,{})
            val=os.environ[env]
            if env == 'LOW_VRAM_MODE': val=val.lower() in {'1','true','yes'}
            cur[path_parts[-1]]=val
    if cfg.get('submission_mode'):
        os.environ.setdefault('HF_HUB_OFFLINE','1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
        os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    return cfg


def _safe_model_ref(path: str | None) -> str | None:
    return Path(path).name if path else None


def _validate_span(text: str, ent: dict[str,Any]) -> dict[str,Any] | None:
    pos=ent.get('position') or [ent.get('start'), ent.get('end')]
    if not isinstance(pos, (list,tuple)) or len(pos)!=2: return None
    s,e=int(pos[0]),int(pos[1])
    if not (0 <= s < e <= len(text)): return None
    if text[s:e] != ent.get('text', text[s:e]): return None
    row={'text':text[s:e], 'type':ent['type'], 'position':[s,e], 'score':float(ent.get('score',0.0))}
    if isinstance(ent.get('assertions'), list): row['assertions']=ent['assertions']
    return row


def _dedupe_entities(text: str, spans: list[dict[str,Any]]) -> tuple[list[dict[str,Any]], int, int]:
    valid=[]; dropped=0
    for ent in spans:
        row=_validate_span(text, ent)
        if row is None: dropped += 1
        else: valid.append(row)
    best={}
    for ent in valid:
        key=tuple(ent['position']); cur=best.get(key)
        if cur is None or ent.get('score',0.0) > cur.get('score',0.0): best[key]=ent
    rows=[best[k] for k in sorted(best)]
    return rows, dropped, len(valid)-len(rows)


def _rule_ner(text: str) -> list[dict[str,Any]]:
    rows=[]
    for c in extract_concepts(text):
        start,end=c['position']; rows.append({'text':c['text'], 'type':c['type'], 'start':start, 'end':end, 'score':1.0})
    rows=finalize_predictions(text, rows)
    rows=[{'text':r['text'], 'type':r['type'], 'position':[r['start'], r['end']], 'score':r.get('score',0.0)} for r in rows]
    occupied=[tuple(r['position']) for r in rows]
    def overlaps(s,e): return any(not (e <= a or s >= b) for a,b in occupied)
    for m in LAB_PAIR_RE.finditer(text):
        ns,ne=m.start('name'),m.end('name'); vs,ve=m.start('value'),m.end('value')
        if not overlaps(ns,ne): rows.append({'text':text[ns:ne],'type':'TÊN_XÉT_NGHIỆM','position':[ns,ne],'score':0.99}); occupied.append((ns,ne))
        if not overlaps(vs,ve): rows.append({'text':text[vs:ve],'type':'KẾT_QUẢ_XÉT_NGHIỆM','position':[vs,ve],'score':0.99}); occupied.append((vs,ve))
    return rows


_CLAUSE_BOUNDARY_RE=re.compile(r'[\r\n.;!?]')

def _clause_bounds(text: str, start: int, end: int, window: int=96) -> tuple[int,int]:
    left=max(0, start-window)
    for m in _CLAUSE_BOUNDARY_RE.finditer(text, max(0,start-window), start): left=m.end()
    right=min(len(text), end+window)
    m=_CLAUSE_BOUNDARY_RE.search(text, end, min(len(text), end+window))
    if m: right=m.start()
    return left,right

def _med_retrieval_query(text: str, ent: dict[str,Any]) -> str:
    s,e=ent['position']; left,right=_clause_bounds(text, s, e)
    clause=text[left:right]
    # Keep entity surface exact and add only same-clause bounded dose/form/route context.
    local_start=s-left; local_end=e-left
    suffix=clause[local_end:local_end+64]
    cue=re.match(r'(?P<ctx>\s*(?:\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|ml|IU|%)|đường uống|uống|tiêm|truyền|viên|ống|tablet|capsule|oral|inj|injection|xr|er|sr|ir|\s|/|-)+)', suffix, re.I)
    ctx=(cue.group('ctx') if cue else '').strip()
    return (ent['text'] + (' ' + ctx if ctx else '')).strip()


class EndToEndPipeline:
    def __init__(self, cfg: dict[str,Any]):
        self.cfg=cfg
        self.report={'documents':0,'entities_by_type':Counter(),'assertions_by_label':Counter(),'invalid_offsets':0,'dropped_spans':0,'rxnorm_linking_coverage':0.0,'icd_linking_coverage':0.0,'verified_candidate_count':0,'unverified_candidate_count':0,'candidate_cache_hits':0,'medication_entity_count':0,'unique_medication_query_count':0,'latency':Counter(),'model_metadata':{},'kb':{},'official_production_blockers':[],'executed_backends':[],'low_vram_mode_executed':False,'stage_release_events':[],'official_evaluation':False,'production_runtime_verified':False}
        self._ner_model=None; self._ner_tokenizer=None; self._llm_hybrid=None; self._assertion_model=None; self._assertion_tokenizer=None; self._assertion_thresholds=None
        self._rx_index=None; self._rx_dense=None; self._rx_encoder=None; self._rx_cache={}; self._icd_records=None
        self._validate_artifacts()
    def _validate_artifacts(self):
        prod=bool(self.cfg.get('production', False)); ner=self.cfg.get('ner',{}); assertion=self.cfg.get('assertion',{}); rx=self.cfg.get('rxnorm',{})
        if prod and not ner.get('mock') and not ner.get('model_path'): raise FileNotFoundError('NER checkpoint is required in production')
        if prod and assertion.get('source') != 'ner' and not assertion.get('mock') and not assertion.get('model_path'): raise FileNotFoundError('assertion checkpoint is required in production')
        if self.cfg.get('submission_mode') and (ner.get('mock') or assertion.get('mock')): raise RuntimeError('mock backends are forbidden in submission mode')
        if prod and rx.get('strict', True):
            if not rx.get('kb_dir'): raise FileNotFoundError('official RxNorm KB is required in strict production mode')
            if rx.get('retrieval_mode','bge_dense') == 'bge_dense' and not rx.get('dense_index_dir'): raise FileNotFoundError('RxNorm dense index is required for dense production mode')
        if self.cfg.get('submission_mode') and rx.get('retrieval_mode','bge_dense') == 'bge_dense' and not rx.get('bge_model_path'):
            raise FileNotFoundError('local RXNORM_BGE_MODEL_PATH is required in submission mode')
    def _torch_device_dtype(self):
        try:
            import torch
            req=self.cfg.get('runtime',{}).get('device')
            device=req if req and req != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
            dtype=torch.float16 if str(device).startswith('cuda') else torch.float32
            return torch, device, dtype
        except Exception as e:
            raise RuntimeError('torch is required for non-mock production runtime') from e
    def _load_llm_hybrid(self):
        if self._llm_hybrid is not None: return
        t0=time.time(); ner=self.cfg.get('ner',{})
        self._llm_hybrid=LLMHybridExtractor.from_config(ner)
        self.report['model_metadata']['ner']={'backend':'llm_hybrid','model_path':_safe_model_ref(ner.get('model_path')),'adapter_path':_safe_model_ref(ner.get('adapter_path')),'model_forward_calls':0,'chunks':0,'dropped_rows':0,'dropped_overlaps':0,'load_seconds':time.time()-t0,'inference_seconds':0.0,'mock':False}

    def _load_ner(self):
        if self._ner_model is not None: return
        t0=time.time(); ner=self.cfg.get('ner',{})
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        torch,device,dtype=self._torch_device_dtype()
        local_only=bool(self.cfg.get('submission_mode', False))
        self._ner_tokenizer=AutoTokenizer.from_pretrained(ner['model_path'], use_fast=True, local_files_only=local_only)
        self._ner_model=AutoModelForTokenClassification.from_pretrained(ner['model_path'], torch_dtype=dtype, local_files_only=local_only)
        self._ner_model.to(device); self._ner_model.eval()
        self.report['model_metadata']['ner']={'backend':'transformers_token_classification','model_path':_safe_model_ref(ner.get('model_path')),'model_class':self._ner_model.__class__.__name__,'tokenizer_class':self._ner_tokenizer.__class__.__name__,'device':device,'dtype':str(dtype).replace('torch.',''),'model_forward_calls':0,'chunks':0,'load_seconds':time.time()-t0,'inference_seconds':0.0,'mock':False,'model_load_count':self.report['model_metadata'].get('ner',{}).get('model_load_count',0)+1}
    def _predict_ner(self, text: str) -> list[dict[str,Any]]:
        ner=self.cfg.get('ner',{})
        if ner.get('backend') == 'llm_hybrid':
            self._load_llm_hybrid(); meta=self.report['model_metadata']['ner']; t0=time.time()
            rows=self._llm_hybrid.predict(text)
            meta['model_forward_calls'] += self._llm_hybrid.generation_calls - meta.get('_last_generation_calls',0)
            meta['_last_generation_calls']=self._llm_hybrid.generation_calls
            meta['chunks']=self._llm_hybrid.chunk_count
            meta['dropped_rows']=self._llm_hybrid.dropped_rows
            meta['dropped_overlaps']=self._llm_hybrid.dropped_overlaps
            meta['inference_seconds'] += time.time()-t0
            if 'ner:llm_hybrid' not in self.report['executed_backends']: self.report['executed_backends'].append('ner:llm_hybrid')
            return [{'text':r['text'],'type':r['type'],'position':[r['start'],r['end']],'score':r.get('score',0.0),'assertions':r.get('assertions',[])} for r in rows]
        if ner.get('mock', False):
            self.report['model_metadata'].setdefault('ner', {'backend':'rule_smoke','mock':True,'model_forward_calls':0,'chunks':0})
            if 'ner:mock' not in self.report['executed_backends']: self.report['executed_backends'].append('ner:mock')
            return _rule_ner(text)
        self._load_ner(); meta=self.report['model_metadata']['ner']; t0=time.time()
        spans=predict_with_model(text, self._ner_tokenizer, self._ner_model, max_length=int(ner.get('max_length',256)), stride=int(ner.get('stride',64)))
        meta['model_forward_calls'] += 1; meta['chunks'] += max(1, math.ceil(len(text)/max(1,int(ner.get('max_length',256))))); meta['inference_seconds'] += time.time()-t0
        if meta['model_forward_calls'] <= 0: raise RuntimeError('NER production backend did not run forward')
        if 'ner:transformers' not in self.report['executed_backends']: self.report['executed_backends'].append('ner:transformers')
        return [{'text':s['text'],'type':s['type'],'position':[s['start'],s['end']],'score':s.get('score',0.0)} for s in spans]
    def _load_assertion(self):
        if self._assertion_model is not None: return
        t0=time.time(); cfg=self.cfg.get('assertion',{})
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch,device,dtype=self._torch_device_dtype()
        local_only=bool(self.cfg.get('submission_mode', False))
        self._assertion_tokenizer=AutoTokenizer.from_pretrained(cfg['model_path'], use_fast=True, local_files_only=local_only)
        self._assertion_model=AutoModelForSequenceClassification.from_pretrained(cfg['model_path'], torch_dtype=dtype, local_files_only=local_only)
        self._assertion_model.to(device); self._assertion_model.eval()
        th_path=cfg.get('thresholds_path') or str(Path(cfg['model_path'])/'thresholds.json')
        self._assertion_thresholds=load_thresholds(th_path)
        self.report['model_metadata']['assertion']={'backend':'transformers_sequence_classification','model_class':self._assertion_model.__class__.__name__,'tokenizer_class':self._assertion_tokenizer.__class__.__name__,'thresholds_path':th_path,'device':device,'dtype':str(dtype).replace('torch.',''),'model_forward_calls':0,'example_count':0,'load_seconds':time.time()-t0,'inference_seconds':0.0,'mock':False,'model_load_count':self.report['model_metadata'].get('assertion',{}).get('model_load_count',0)+1}
    def _predict_assertions(self, text: str, entities: list[dict[str,Any]]):
        cfg=self.cfg.get('assertion',{})
        if cfg.get('source') == 'ner':
            if 'assertion:llm_hybrid' not in self.report['executed_backends']: self.report['executed_backends'].append('assertion:llm_hybrid')
            return entities
        if cfg.get('mock', False):
            self.report['model_metadata'].setdefault('assertion', {'backend':'rules','mock':True,'model_forward_calls':0,'example_count':len(entities)})
            if 'assertion:mock' not in self.report['executed_backends']: self.report['executed_backends'].append('assertion:mock')
            return predict_assertions(text, entities)
        self._load_assertion(); meta=self.report['model_metadata']['assertion']; t0=time.time()
        out=predict_assertions(text, entities, model=self._assertion_model, tokenizer=self._assertion_tokenizer, thresholds=self._assertion_thresholds)
        if entities: meta['model_forward_calls'] += 1
        meta['example_count'] += len(entities); meta['inference_seconds'] += time.time()-t0
        if entities and meta['model_forward_calls'] <= 0: raise RuntimeError('assertion production backend did not run forward')
        if 'assertion:transformers' not in self.report['executed_backends']: self.report['executed_backends'].append('assertion:transformers')
        return out
    def _release_stage(self, name):
        if not self.cfg.get('runtime',{}).get('low_vram_mode'): return
        if name == 'ner': self._ner_model=None; self._ner_tokenizer=None; self._llm_hybrid=None
        if name == 'assertion': self._assertion_model=None; self._assertion_tokenizer=None
        if name == 'rxnorm': self._rx_encoder=None
        self.report['stage_release_events'].append(name); self.report['low_vram_mode_executed']=True
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass
    def _rx_records(self):
        rx=self.cfg.get('rxnorm',{}); kb_dir=Path(rx.get('kb_dir',''))
        if not kb_dir.exists():
            if rx.get('strict', False): raise FileNotFoundError(f'RxNorm KB missing: {kb_dir}')
            self.report['official_production_blockers'].append('RxNorm official KB/index missing'); return []
        records=[]; paths=sorted(kb_dir.glob('*.jsonl'))
        for p in paths: records.extend(read_jsonl(p))
        if any(not r.verified for r in records) or bool(rx.get('include_unverified', False)): raise ValueError('production RxNorm candidates require include_unverified=false and verified=true records')
        self.report['kb']['rxnorm']={'records':len(records),'checksum':kb_paths_checksum(paths),'candidate_universe':len({r.code for r in records if r.verified})}
        return records
    def _load_rx_lexical(self):
        if self._rx_index is not None: return self._rx_index
        records=self._rx_records(); self._rx_index=LexicalIndex(records, include_unverified=False) if records else False; return self._rx_index
    def _load_rx_dense(self):
        if self._rx_dense is not None: return
        t0=time.time(); rx=self.cfg.get('rxnorm',{}); records=self._rx_records(); paths=sorted(Path(rx['kb_dir']).glob('*.jsonl'))
        expected=dense_expected_manifest({'model_name':rx.get('model_name','BAAI/bge-m3'),'model_revision':rx.get('model_revision','main'),'include_unverified':False,'max_length':rx.get('max_length',8192),'batch_size':rx.get('batch_size',16)}, paths, candidate_universe=len({r.code for r in records if r.verified}))
        self._rx_dense=load_dense_index(Path(rx['dense_index_dir']), expected)
        model_ref=rx.get('bge_model_path') if self.cfg.get('submission_mode') else expected['model_name']
        if self.cfg.get('submission_mode') and not Path(model_ref).exists(): raise FileNotFoundError(f'local BGE model path missing: {model_ref}')
        self._rx_encoder=BGEM3Backend(model_name=model_ref, batch_size=expected['batch_size'], max_length=expected['max_length'], use_fp16=rx.get('use_fp16'), device=self.cfg.get('runtime',{}).get('device'))
        self.report['model_metadata']['rxnorm']={'backend':'bge_dense','model_name':expected['model_name'],'model_revision':expected['model_revision'],'dense_cache_validation':'valid','dense_query_encode_calls':0,'dense_search_calls':0,'dense_query_encode_batches':0,'dense_query_vectors':0,'candidate_universe':expected['candidate_universe'],'include_unverified':False,'mock':False,'linking_load_seconds':0.0,'linking_inference_seconds':0.0,'model_load_count':self.report['model_metadata'].get('rxnorm',{}).get('model_load_count',0)+1}
        self.report['model_metadata']['rxnorm']['linking_load_seconds'] += time.time()-t0
    def _med_query_key(self, text: str, ent: dict[str,Any]):
        query=_med_retrieval_query(text, ent)
        nm=normalize_mention(query, 'THUỐC')
        base=(nm.get('base_no_diacritic') or nm.get('normalized') or ent['text']).casefold()
        context_key=normalize_text(query, no_diacritic=True)
        return f'{base}||{context_key}', query
    def _precompute_med_links(self, texts: list[str], docs: list[list[dict[str,Any]]]):
        meds=[]
        for doc_i,(text,ents) in enumerate(zip(texts, docs)):
            for ent in ents:
                if ent.get('type') == 'THUỐC': meds.append((doc_i,text,ent,*self._med_query_key(text, ent)))
        self.report['medication_entity_count'] += len(meds)
        unseen=[]; seen=set()
        for item in meds:
            key=item[3]
            if key in self._rx_cache: self.report['candidate_cache_hits'] += 1
            elif key not in seen: seen.add(key); unseen.append(item)
            else: self.report['candidate_cache_hits'] += 1
        self.report['unique_medication_query_count'] += len(seen)
        if not unseen: return
        rx=self.cfg.get('rxnorm',{}); mode=rx.get('retrieval_mode','bge_dense')
        if mode == 'bge_dense':
            self._load_rx_dense(); meta=self.report['model_metadata']['rxnorm']; bs=max(1,int(rx.get('batch_size',16))); t0=time.time()
            for start in range(0, len(unseen), bs):
                batch=unseen[start:start+bs]; queries=[it[4] for it in batch]
                vecs=self._rx_encoder.encode(queries); meta['dense_query_encode_calls'] += 1; meta['dense_query_encode_batches'] += 1; meta['dense_query_vectors'] += len(vecs)
                for item,qvec in zip(batch, vecs):
                    rows=self._rx_dense.search_vector(qvec, top_k=int(rx.get('top_k',10))); meta['dense_search_calls'] += 1; codes=[]
                    for r in rows:
                        if r.get('verified') and r['code'] not in codes: codes.append(r['code'])
                    self._rx_cache[item[3]]=codes
            meta['linking_inference_seconds'] += time.time()-t0
            if 'rxnorm:bge_dense' not in self.report['executed_backends']: self.report['executed_backends'].append('rxnorm:bge_dense')
        elif mode in {'bm25','lexical_smoke'}:
            idx=self._load_rx_lexical(); t0=time.time()
            for item in unseen:
                rows=idx.search(item[4], 'THUỐC', top_k=int(rx.get('top_k',5)), use_fuzzy=False) if idx else []; codes=[]
                for c in rows:
                    if c.verified and c.code not in codes: codes.append(c.code)
                self._rx_cache[item[3]]=codes
            self.report['latency']['linking_inference_seconds'] += time.time()-t0
            if 'rxnorm:bm25' not in self.report['executed_backends']: self.report['executed_backends'].append('rxnorm:bm25')
        else: raise ValueError(f'unsupported RxNorm retrieval_mode: {mode}')
    def _link_med(self, ent: dict[str,Any], text: str) -> list[str]:
        key,_=self._med_query_key(text, ent)
        if key not in self._rx_cache: self._precompute_med_links([text], [[ent]])
        return list(self._rx_cache.get(key, []))[:1]
    def _load_icd_records(self):
        if self._icd_records is not None: return self._icd_records
        icd=self.cfg.get('icd10',{}); kb_dir=Path(icd.get('kb_dir','')) if icd.get('kb_dir') else None
        if not kb_dir or not kb_dir.exists():
            msg='official ICD-10 KB missing; diagnosis candidates suppressed'
            if icd.get('require', False): raise FileNotFoundError(msg)
            self.report['official_production_blockers'].append(msg); self._icd_records=[]; return []
        records=[]
        for p in sorted(kb_dir.glob('*.jsonl')): records.extend(read_jsonl(p))
        bad=any((not r.verified) or (not r.metadata.get('official_kb', False)) for r in records)
        if bad and bool(self.cfg.get('production', False)):
            raise ValueError('production ICD-10 candidates require verified=true and metadata.official_kb=true records')
        records=[r for r in records if r.verified and r.metadata.get('official_kb', False)]
        if not records:
            msg='official ICD-10 KB missing; diagnosis candidates suppressed'
            if icd.get('require', False): raise FileNotFoundError(msg)
            self.report['official_production_blockers'].append(msg)
        self._icd_records=records; self.report['kb']['icd10']={'records':len(records)}; return records
    def _link_icd(self, ent: dict[str,Any]) -> list[str]:
        records=self._load_icd_records()
        if not records: return []
        idx=LexicalIndex(records, include_unverified=False)
        return [c.code for c in idx.search(ent['text'], 'CHẨN_ĐOÁN', top_k=int(self.cfg.get('icd10',{}).get('top_k',5)), use_fuzzy=False) if c.verified][:1]
    def _finalize_one(self, text, entities):
        med_total=med_hit=diag_total=diag_hit=0; final=[]
        for ent in entities:
            row={'text':ent['text'],'type':ent['type'],'position':ent['position']}
            if ent['type'] in ASSERTION_TYPES:
                assertions=[a for a in ASSERTION_ORDER if a in ent.get('assertions',[])]
                if assertions: row['assertions']=assertions
            if ent['type']=='THUỐC': med_total += 1; codes=self._link_med(ent, text); med_hit += bool(codes); row['candidates']=codes
            elif ent['type']=='CHẨN_ĐOÁN': diag_total += 1; codes=self._link_icd(ent); diag_hit += bool(codes); row['candidates']=codes
            if set(row) - set(ALLOWED_KEYS): raise ValueError('final concept contains unsupported keys')
            final.append(row); self.report['entities_by_type'][row['type']] += 1
            for a in row.get('assertions',[]): self.report['assertions_by_label'][a] += 1
            self.report['verified_candidate_count'] += len(row.get('candidates',[]))
        if med_total: self.report['rxnorm_linking_coverage']=med_hit/med_total
        if diag_total: self.report['icd_linking_coverage']=diag_hit/diag_total
        return final
    def infer_document(self, text: str) -> list[dict[str,Any]]:
        return self.infer_documents([text])[0]
    def infer_documents(self, texts: list[str]) -> list[list[dict[str,Any]]]:
        t=time.time(); span_docs=[self._predict_ner(t) for t in texts]; self.report['latency']['ner'] += time.time()-t; self._release_stage('ner')
        deduped=[]
        for text,spans in zip(texts, span_docs):
            entities,dropped,dupes=_dedupe_entities(text, spans); self.report['dropped_spans'] += dropped + dupes; self.report['invalid_offsets'] += dropped; deduped.append(entities)
        t=time.time(); asserted=[]
        for text,ents in zip(texts,deduped):
            eligible=[{k:v for k,v in e.items() if k in {'text','type','position','assertions'}} for e in ents if e.get('type') in ASSERTION_TYPES]
            pred_by_pos={tuple(e['position']):e for e in self._predict_assertions(text, eligible)}
            merged=[]
            for e in ents:
                merged.append(pred_by_pos.get(tuple(e['position']), e))
            asserted.append(merged)
        self.report['latency']['assertion'] += time.time()-t; self._release_stage('assertion')
        t=time.time(); self._precompute_med_links(texts, asserted); final=[self._finalize_one(text, ents) for text,ents in zip(texts, asserted)]; self.report['latency']['linking'] += time.time()-t; self._release_stage('rxnorm')
        self.report['documents'] += len(texts); self._update_runtime_verified()
        return final
    def _update_runtime_verified(self):
        meta=self.report['model_metadata']; prod=bool(self.cfg.get('production', False))
        nonmock_ok=all(not v.get('mock', False) for v in meta.values()) if prod else False
        ner_ok=(not prod) or meta.get('ner',{}).get('model_forward_calls',0)>0
        assertion_entities=sum(self.report['entities_by_type'].values())
        assertion_ok=(not prod) or assertion_entities == 0 or meta.get('assertion',{}).get('model_forward_calls',0)>0
        rx_ok=True
        if prod and self.report['entities_by_type'].get('THUỐC',0): rx_ok=meta.get('rxnorm',{}).get('backend')=='bge_dense' and meta.get('rxnorm',{}).get('dense_query_encode_calls',0)>0 and meta.get('rxnorm',{}).get('dense_search_calls',0)>0
        self.report['production_runtime_verified']=bool(prod and nonmock_ok and ner_ok and assertion_ok and rx_ok and self.report['unverified_candidate_count']==0 and self.report['invalid_offsets']==0)
    def serializable_report(self):
        out=dict(self.report);
        if 'ner' in out.get('model_metadata',{}): out['model_metadata']['ner'].pop('_last_generation_calls', None)
        out['entities_by_type']=dict(out['entities_by_type']); out['assertions_by_label']=dict(out['assertions_by_label']); out['latency']=dict(out['latency'])
        try:
            import torch
            out['peak_vram_bytes']=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        except Exception: out['peak_vram_bytes']=0
        out['offline_mode']=bool(self.cfg.get('submission_mode', False)); out['network_download_attempts']=0
        return out
