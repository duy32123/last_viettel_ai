from __future__ import annotations
import math, difflib, json, time, sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any
from pathlib import Path
from src.data.kb_schema import KBRecord, read_jsonl, alias_collisions, file_sha256
from .normalization import normalize_mention, normalize_text
ROUTE={'CHẨN_ĐOÁN':'ICD-10','THUỐC':'RxNorm'}
@dataclass
class Candidate:
    code:str; terminology:str; canonical_name:str; score:float; lexical_score:float; embedding_score:None; reranker_score:None; verified:bool; source:str; version:str; match_method:str
    def to_dict(self): return self.__dict__.copy()

def _tokens(s): return [t for t in normalize_text(s, True).split() if t]
class LexicalIndex:
    def __init__(self, records:list[KBRecord], include_unverified: bool=False):
        t0=time.time(); self.records=[r for r in records if r.verified or include_unverified]; self.names=[]; self.exact={}; self.df=Counter(); self.postings=defaultdict(list); self.build_seconds=0.0
        for i,r in enumerate(self.records):
            for name in [r.canonical_name,*r.aliases]:
                for n in {normalize_text(name), normalize_text(name, True)}:
                    self.exact.setdefault((r.terminology,n), []).append((i,name))
                dtok=Counter(_tokens(name)); toks=set(dtok); self.df.update(toks); self.names.append((i,name,dtok))
                for tok in toks: self.postings[(r.terminology,tok)].append((i,name,dtok[tok]))
        self.build_seconds=time.time()-t0; self.index_size={'records':len(self.records),'name_entries':len(self.names),'postings':sum(len(v) for v in self.postings.values()),'exact_keys':len(self.exact)}
    @classmethod
    def from_paths(cls, paths):
        recs=[]
        for p in paths: recs.extend(read_jsonl(p))
        return cls(recs)
    def search(self, mention:str, entity_type:str, top_k:int=5, use_fuzzy:bool=True):
        term=ROUTE.get(entity_type)
        if not term: return []
        nm=normalize_mention(mention, entity_type); queries=[nm['base'],nm['normalized'],nm['base_no_diacritic'],nm['no_diacritic']]
        hits=[]; seen=set(); drug_features=nm.get('features',{})
        def add(idx,score,method):
            r=self.records[idx]; key=(r.terminology,r.version,r.code)
            if key in seen: return
            seen.add(key); cand=Candidate(r.code,r.terminology,r.canonical_name,score,score,None,None,r.verified,r.source,r.version,method); cand.tty=r.metadata.get('TTY'); cand.specificity=r.metadata.get('specificity'); cand.matched_alias=name; hits.append(cand)
        for q in queries:
            for idx,name in self.exact.get((term,q),[]): add(idx,1.0,'exact' if normalize_text(self.records[idx].canonical_name) == q else 'alias')
        if len(hits)<top_k:
            qtok=Counter(_tokens(nm['no_diacritic'] if term == 'RxNorm' and drug_features else nm['base_no_diacritic'])); N=max(1,len(self.records)); scores=[]
            acc={}
            best_name={}
            for tok,cnt in qtok.items():
                idf=math.log((N-self.df[tok]+0.5)/(self.df[tok]+0.5)+1)
                for idx,name,tf in self.postings.get((term,tok),[]):
                    acc[idx]=acc.get(idx,0.0)+idf*tf
                    best_name.setdefault(idx,name)
            for idx,s in acc.items():
                if s>0: scores.append((s,idx,best_name[idx]))
            for s,idx,name in sorted(scores, key=lambda x:(-x[0], self.records[x[1]].terminology,self.records[x[1]].code)): add(idx, min(0.89,s/(s+1)),'bm25')
        if use_fuzzy and len(hits)<top_k:
            q=nm['base_no_diacritic']
            scores=[]
            for idx,name,_ in self.names:
                r=self.records[idx]
                if r.terminology==term: scores.append((difflib.SequenceMatcher(None,q,normalize_text(name,True)).ratio(),idx))
            for s,idx in sorted(scores, key=lambda x:(-x[0], self.records[x[1]].code)):
                if s>=0.82: add(idx, s*0.75, 'fuzzy')

        def med_key(c):
            if term != 'RxNorm': return (-c.score,c.terminology,c.code)
            spec=getattr(c,'specificity',None); has_detail=bool(drug_features.get('strength') or drug_features.get('dose_form'))
            pref=0
            if has_detail and spec=='product': pref=-0.08
            elif not has_detail and spec=='ingredient': pref=-0.08
            elif not has_detail and spec=='product': pref=0.08
            elif spec=='brand': pref=0.0
            return (pref,-c.score,c.code)

        rows=sorted(hits, key=med_key)[:top_k]
        for i,c in enumerate(rows,1):
            c.rank=i; c.retrieval_method=c.match_method
        return rows


def kb_paths_checksum(paths):
    h={}
    for p in sorted(Path(x) for x in paths): h[p.name]=file_sha256(p)
    return h

def _record_payload(r: KBRecord):
    return {'code':r.code,'canonical_name':r.canonical_name,'aliases':r.aliases,'terminology':r.terminology,'version':r.version,'source':r.source,'verified':r.verified,'metadata':r.metadata,'semantic_type':r.semantic_type,'language':r.language}

def save_lexical_index(index: LexicalIndex, out_dir: Path, kb_paths=None, manifest_extra=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest={'index_type':'lexical_bm25_inverted','schema_version':1,'build_seconds':index.build_seconds,'index_size':index.index_size,'kb_checksum':kb_paths_checksum(kb_paths or []),'peak_memory_bytes':None,'candidate_universe':len(index.records)}
    if manifest_extra: manifest.update(manifest_extra)
    (out_dir/'lexical_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    payload={'records':[_record_payload(r) for r in index.records], 'names':[(i,n,dict(c)) for i,n,c in index.names], 'exact':{f'{k[0]}\t{k[1]}':v for k,v in index.exact.items()}, 'df':dict(index.df), 'postings':{f'{k[0]}\t{k[1]}':v for k,v in index.postings.items()}, 'index_size':index.index_size}
    (out_dir/'lexical_index.json').write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    (out_dir/'lexical_postings_summary.json').write_text(json.dumps(index.index_size, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest

def load_lexical_index(kb_dir: Path, include_unverified=False, expected_kb_checksum:dict|None=None, index_dir:Path|None=None):
    t0=time.time(); paths=sorted(Path(kb_dir).glob('*.jsonl')); checksum=kb_paths_checksum(paths)
    if expected_kb_checksum is not None and checksum != expected_kb_checksum: raise ValueError('stale lexical index cache')
    idx_dir=Path(index_dir) if index_dir else Path(kb_dir)
    manifest_path=idx_dir/'lexical_manifest.json'; payload_path=idx_dir/'lexical_index.json'
    if not manifest_path.exists() or not payload_path.exists(): raise FileNotFoundError(f'persisted lexical index missing: {idx_dir}')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if expected_kb_checksum is not None and manifest.get('kb_checksum') != expected_kb_checksum: raise ValueError('stale lexical index cache manifest')
    if manifest.get('schema_version') != 1: raise ValueError('unsupported lexical index schema')
    payload=json.loads(payload_path.read_text(encoding='utf-8'))
    obj=LexicalIndex.__new__(LexicalIndex)
    obj.records=[KBRecord(d['code'],d['canonical_name'],d.get('aliases',[]),d['terminology'],d['version'],d['source'],d.get('verified',False),d.get('metadata',{}),d.get('semantic_type',''),d.get('language','en')) for d in payload['records']]
    if not include_unverified and any(not r.verified for r in obj.records): raise ValueError('persisted production lexical index contains unverified records')
    obj.names=[(i,n,Counter(c)) for i,n,c in payload['names']]
    obj.exact={tuple(k.split('\t',1)):v for k,v in payload['exact'].items()}
    obj.df=Counter(payload['df'])
    obj.postings=defaultdict(list, {tuple(k.split('\t',1)):v for k,v in payload['postings'].items()})
    obj.index_size=payload.get('index_size', manifest.get('index_size',{})); obj.build_seconds=manifest.get('build_seconds',0); obj.loaded_from_cache=True; obj.index_load_seconds=time.time()-t0
    return obj

def report_records(records:list[KBRecord])->dict[str,Any]:
    by=Counter((r.terminology,r.version,r.source) for r in records); tty=Counter(r.metadata.get('TTY') for r in records if r.metadata.get('TTY'))
    return {'rows_by_terminology_version_source':{str(k):v for k,v in by.items()}, 'rows_by_tty':dict(tty), 'verified':sum(r.verified for r in records), 'unverified':sum(not r.verified for r in records), 'unique_codes':len({(r.terminology,r.version,r.code) for r in records}), 'alias_count':sum(len(r.aliases) for r in records), 'alias_collisions':alias_collisions(records), 'empty_canonical_names':sum(not r.canonical_name for r in records), 'duplicates':len(records)-len({(r.terminology,r.version,r.code) for r in records})}

def expand_relationship_candidates(candidates:list[dict], records:list[KBRecord], enabled:bool=False, max_related:int=5)->list[dict]:
    if not enabled: return candidates
    by_code={r.code:r for r in records}; seen={c['code'] for c in candidates}; out=list(candidates)
    for cand in candidates:
        rec=by_code.get(cand['code'])
        if not rec: continue
        for rel in rec.metadata.get('relationships',[])[:max_related]:
            code=rel.get('target_rxcui')
            if not code or code in seen or code not in by_code: continue
            r=by_code[code]; seen.add(code); out.append({'code':r.code,'terminology':r.terminology,'canonical_name':r.canonical_name,'score':0.0,'rank':len(out)+1,'retrieval_method':'rxnrel_expansion','matched_alias':r.canonical_name,'verified':r.verified,'source':r.source,'version':r.version,'tty':r.metadata.get('TTY'),'relationship_provenance':rel})
    return out
