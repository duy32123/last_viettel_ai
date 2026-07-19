from __future__ import annotations
import hashlib, json, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
from src.data.kb_schema import KBRecord, file_sha256
from src.linking.normalization import normalize_text
from src.linking.retrieval import LexicalIndex

BGE_M3_MODEL='BAAI/bge-m3'

def l2_normalize(vec):
    norm=math.sqrt(sum(float(x)*float(x) for x in vec)) or 1.0
    return [float(x)/norm for x in vec]

def dot(a,b): return sum(float(x)*float(y) for x,y in zip(a,b))

class BGEM3Backend:
    def __init__(self, model_name=BGE_M3_MODEL, batch_size=16, max_length=8192, use_fp16=None, device=None):
        self.model_name=model_name; self.batch_size=batch_size; self.max_length=max_length; self.use_fp16=use_fp16; self.device=device; self.model=None
    def load(self):
        from FlagEmbedding import BGEM3FlagModel
        if self.use_fp16 is None:
            try:
                import torch; fp16=bool(torch.cuda.is_available())
            except Exception: fp16=False
        else: fp16=bool(self.use_fp16)
        self.model=BGEM3FlagModel(self.model_name, use_fp16=fp16, device=self.device)
    def encode(self, texts):
        if self.model is None: self.load()
        out=self.model.encode(list(texts), batch_size=self.batch_size, max_length=self.max_length, return_dense=True, return_sparse=False, return_colbert_vecs=False)
        return [l2_normalize(v) for v in out['dense_vecs']]

class MockDenseEncoder:
    def __init__(self, dim=4): self.dim=dim
    def encode(self, texts):
        vecs=[]
        for t in texts:
            h=hashlib.sha256(normalize_text(t, True).encode()).digest(); vals=[(h[i]/255.0) for i in range(self.dim)]
            vecs.append(l2_normalize(vals))
        return vecs

@dataclass
class AliasEntry:
    alias: str; normalized_alias: str; code: str; canonical_name: str; terminology: str; verified: bool; source: str; version: str

def alias_entries(records:list[KBRecord], include_unverified=False):
    entries=[]
    for r in records:
        if not r.verified and not include_unverified: continue
        for alias in [r.canonical_name,*r.aliases]:
            entries.append(AliasEntry(alias, normalize_text(alias, True), r.code, r.canonical_name, r.terminology, r.verified, r.source, r.version))
    return entries

class DenseAliasIndex:
    def __init__(self, entries:list[AliasEntry], vectors:list[list[float]], manifest:dict[str,Any]|None=None):
        self.entries=entries; self.vectors=[l2_normalize(v) for v in vectors]; self.manifest=manifest or {}
    @classmethod
    def build(cls, records, encoder, include_unverified=False, manifest=None):
        entries=alias_entries(records, include_unverified); vecs=encoder.encode([e.alias for e in entries]) if entries else []
        return cls(entries, vecs, manifest)
    def search(self, query, top_k=10):
        q=MockDenseEncoder(dim=len(self.vectors[0]) if self.vectors else 4).encode([query])[0]
        return self.search_vector(q, top_k)
    def search_with_encoder(self, query, encoder, top_k=10):
        return self.search_vector(encoder.encode([query])[0], top_k)
    def search_vector(self, qvec, top_k=10):
        best={}
        for e,v in zip(self.entries,self.vectors):
            s=dot(qvec,v); cur=best.get(e.code)
            if cur is None or s>cur['score'] or (s==cur['score'] and e.alias<cur['matched_alias']):
                best[e.code]={'code':e.code,'canonical_name':e.canonical_name,'terminology':e.terminology,'score':s,'matched_alias':e.alias,'verified':e.verified,'source':e.source,'version':e.version,'retrieval_method':'bge_dense'}
        rows=sorted(best.values(), key=lambda x:(-x['score'], x['code']))[:top_k]
        for i,r in enumerate(rows,1): r['rank']=i
        return rows

def kb_checksum(paths):
    h=hashlib.sha256()
    for p in sorted(Path(x) for x in paths): h.update(file_sha256(p).encode())
    return h.hexdigest()

def save_dense_index(index:DenseAliasIndex, out_dir:Path, manifest:dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/'embeddings.json').write_text(json.dumps(index.vectors), encoding='utf-8')
    (out_dir/'alias_to_code.json').write_text(json.dumps([e.__dict__ for e in index.entries], ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir/'dense_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

def load_dense_index(out_dir:Path, expected:dict):
    manifest=json.loads((out_dir/'dense_manifest.json').read_text())
    for k,v in expected.items():
        if manifest.get(k)!=v: raise ValueError(f'stale dense cache for {k}')
    entries=[AliasEntry(**d) for d in json.loads((out_dir/'alias_to_code.json').read_text())]
    vecs=json.loads((out_dir/'embeddings.json').read_text())
    return DenseAliasIndex(entries, vecs, manifest)

def rrf_fuse(rankings:list[tuple[float,list[str]]], k=60, top_k=10):
    scores={}
    for weight,codes in rankings:
        for rank,code in enumerate(codes,1): scores[code]=scores.get(code,0.0)+weight/(k+rank)
    return [c for c,_ in sorted(scores.items(), key=lambda x:(-x[1],x[0]))[:top_k]]

def metrics_from_ranks(ranks):
    n=len(ranks) or 1
    return {'recall@1':sum(1 for r in ranks if r and r<=1)/n,'recall@5':sum(1 for r in ranks if r and r<=5)/n,'recall@10':sum(1 for r in ranks if r and r<=10)/n,'recall@20':sum(1 for r in ranks if r and r<=20)/n,'mrr':sum(1/r for r in ranks if r)/n}

def tune_rrf(dev_examples, bm25_ranker, dense_ranker, weights=(0.5,1.0,2.0), ks=(10,60)):
    best=None
    for wb in weights:
      for wd in weights:
       for k in ks:
        ranks=[]
        for ex in dev_examples:
            bm=bm25_ranker(ex); de=dense_ranker(ex); fused=rrf_fuse([(wb,bm),(wd,de)], k=k, top_k=20)
            ranks.append(fused.index(ex['positive_code'])+1 if ex['positive_code'] in fused else None)
        m=metrics_from_ranks(ranks); key=(m['recall@10'],m['mrr'],m['recall@1'], -wb, -wd, -k)
        if best is None or key>best[0]: best=(key,{'bm25_weight':wb,'dense_weight':wd,'rrf_k':k,'dev_metrics':m})
    return best[1]
