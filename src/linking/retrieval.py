from __future__ import annotations
import math, difflib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any
from src.data.kb_schema import KBRecord, read_jsonl, alias_collisions
from .normalization import normalize_mention, normalize_text
ROUTE={'CHẨN_ĐOÁN':'ICD-10','THUỐC':'RxNorm'}
@dataclass
class Candidate:
    code:str; terminology:str; canonical_name:str; score:float; lexical_score:float; embedding_score:None; reranker_score:None; verified:bool; source:str; version:str; match_method:str
    def to_dict(self): return self.__dict__.copy()

def _tokens(s): return [t for t in normalize_text(s, True).split() if t]
class LexicalIndex:
    def __init__(self, records:list[KBRecord], include_unverified: bool=False):
        self.records=[r for r in records if r.verified or include_unverified]; self.names=[]; self.exact={}; self.df=Counter()
        for i,r in enumerate(self.records):
            for name in [r.canonical_name,*r.aliases]:
                for n in {normalize_text(name), normalize_text(name, True)}:
                    self.exact.setdefault((r.terminology,n), []).append((i,name))
                toks=set(_tokens(name)); self.df.update(toks); self.names.append((i,name,Counter(_tokens(name))))
    @classmethod
    def from_paths(cls, paths):
        recs=[]
        for p in paths: recs.extend(read_jsonl(p))
        return cls(recs)
    def search(self, mention:str, entity_type:str, top_k:int=5, use_fuzzy:bool=True):
        term=ROUTE.get(entity_type)
        if not term: return []
        nm=normalize_mention(mention, entity_type); queries=[nm['base'],nm['normalized'],nm['base_no_diacritic'],nm['no_diacritic']]
        hits=[]; seen=set()
        def add(idx,score,method):
            r=self.records[idx]; key=(r.terminology,r.version,r.code)
            if key in seen: return
            seen.add(key); hits.append(Candidate(r.code,r.terminology,r.canonical_name,score,score,None,None,r.verified,r.source,r.version,method))
        for q in queries:
            for idx,name in self.exact.get((term,q),[]): add(idx,1.0,'exact' if normalize_text(self.records[idx].canonical_name) == q else 'alias')
        if len(hits)<top_k:
            qtok=Counter(_tokens(nm['base_no_diacritic'])); N=max(1,len(self.records)); scores=[]
            for idx,name,dtok in self.names:
                r=self.records[idx]
                if r.terminology!=term: continue
                s=0.0
                for tok,c in qtok.items():
                    if tok in dtok: s += (math.log((N-self.df[tok]+0.5)/(self.df[tok]+0.5)+1) * dtok[tok])
                if s>0: scores.append((s,idx))
            for s,idx in sorted(scores, key=lambda x:(-x[0], self.records[x[1]].terminology,self.records[x[1]].code)): add(idx, min(0.89,s/(s+1)),'bm25')
        if use_fuzzy and len(hits)<top_k:
            q=nm['base_no_diacritic']
            scores=[]
            for idx,name,_ in self.names:
                r=self.records[idx]
                if r.terminology==term: scores.append((difflib.SequenceMatcher(None,q,normalize_text(name,True)).ratio(),idx))
            for s,idx in sorted(scores, key=lambda x:(-x[0], self.records[x[1]].code)):
                if s>=0.82: add(idx, s*0.75, 'fuzzy')
        return sorted(hits, key=lambda c:(-c.score,c.terminology,c.code))[:top_k]

def report_records(records:list[KBRecord])->dict[str,Any]:
    by=Counter((r.terminology,r.version,r.source) for r in records); tty=Counter(r.metadata.get('TTY') for r in records if r.metadata.get('TTY'))
    return {'rows_by_terminology_version_source':{str(k):v for k,v in by.items()}, 'rows_by_tty':dict(tty), 'verified':sum(r.verified for r in records), 'unverified':sum(not r.verified for r in records), 'unique_codes':len({(r.terminology,r.version,r.code) for r in records}), 'alias_count':sum(len(r.aliases) for r in records), 'alias_collisions':alias_collisions(records), 'empty_canonical_names':sum(not r.canonical_name for r in records), 'duplicates':len(records)-len({(r.terminology,r.version,r.code) for r in records})}
