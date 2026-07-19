from __future__ import annotations
import hashlib, json, time, math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency fallback
    np = None
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

BGE_RERANKER_MODEL='BAAI/bge-reranker-v2-m3'

class BGERerankerBackend:
    def __init__(self, model_name=BGE_RERANKER_MODEL, batch_size=16, max_length=8192, use_fp16=None, device=None):
        self.model_name=model_name; self.batch_size=batch_size; self.max_length=max_length; self.use_fp16=use_fp16; self.device=device; self.model=None
    def load(self):
        from FlagEmbedding import FlagReranker
        if self.use_fp16 is None:
            try:
                import torch; fp16=bool(torch.cuda.is_available())
            except Exception: fp16=False
        else: fp16=bool(self.use_fp16)
        self.model=FlagReranker(self.model_name, use_fp16=fp16, device=self.device)
    def score(self, pairs):
        if self.model is None: self.load()
        rows=list(pairs); out=[]
        for start in range(0, len(rows), max(1,self.batch_size)):
            batch=rows[start:start+max(1,self.batch_size)]
            scores=self.model.compute_score(batch, batch_size=self.batch_size, max_length=self.max_length)
            if isinstance(scores, (float,int)): scores=[scores]
            out.extend(float(x) for x in scores)
        return out

class MockReranker:
    def __init__(self, scores:dict[tuple[str,str],float]|None=None): self.scores=scores or {}; self.calls=[]
    def score(self, pairs):
        rows=list(pairs); self.calls.append(rows)
        return [float(self.scores.get((q,p), len(set(normalize_text(q, True).split()) & set(normalize_text(p, True).split())))) for q,p in rows]

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
        self.matrix = np.asarray(self.vectors, dtype='float32') if np is not None and self.vectors else None
    @classmethod
    def build(cls, records, encoder, include_unverified=False, manifest=None):
        entries=alias_entries(records, include_unverified); vecs=encoder.encode([e.alias for e in entries]) if entries else []
        return cls(entries, vecs, manifest)
    def search(self, query, top_k=10):
        raise RuntimeError('DenseAliasIndex.search requires an explicit encoder or query vector; use search_with_encoder() or search_vector()')
    def search_with_encoder(self, query, encoder, top_k=10):
        return self.search_vector(encoder.encode([query])[0], top_k)
    def _scores(self, qvec):
        if self.matrix is not None:
            return [float(x) for x in (self.matrix @ np.asarray(qvec, dtype='float32'))]
        return [dot(qvec, v) for v in self.vectors]
    def search_vector(self, qvec, top_k=10):
        if self.vectors and len(qvec) != len(self.vectors[0]): raise ValueError(f'query/index dimension mismatch: {len(qvec)} != {len(self.vectors[0])}')
        norm=math.sqrt(sum(float(x)*float(x) for x in qvec)) or 0.0
        if not all(math.isfinite(float(x)) for x in qvec) or abs(norm-1.0)>1e-4: raise ValueError('query vector must be finite and L2-normalized')
        best={}
        for e,s in zip(self.entries,self._scores(qvec)):
            cur=best.get(e.code)
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

    if np is not None:
        np.save(out_dir/'embeddings.npy', np.asarray(index.vectors, dtype='float32'))
    else:
        (out_dir/'embeddings.json').write_text(json.dumps(index.vectors), encoding='utf-8')
    (out_dir/'alias_to_code.json').write_text(json.dumps([e.__dict__ for e in index.entries], ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir/'dense_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

def load_dense_index(out_dir:Path, expected:dict):
    if not (out_dir/'dense_manifest.json').exists(): raise FileNotFoundError(f'dense index missing: {out_dir}')
    manifest=json.loads((out_dir/'dense_manifest.json').read_text())
    for k,v in expected.items():
        if manifest.get(k)!=v: raise ValueError(f'stale dense cache for {k}')
    entries=[AliasEntry(**d) for d in json.loads((out_dir/'alias_to_code.json').read_text())]

    if (out_dir/'embeddings.npy').exists() and np is not None:
        vecs=np.load(out_dir/'embeddings.npy').astype('float32').tolist()
    else:
        vecs=json.loads((out_dir/'embeddings.json').read_text())
    dim=manifest.get('dimension') or (len(vecs[0]) if vecs else 0)
    if vecs and any(len(v)!=dim for v in vecs): raise ValueError('dense cache vector dimension mismatch')
    if vecs and any(abs(math.sqrt(sum(float(x)*float(x) for x in v))-1.0)>1e-4 for v in vecs): raise ValueError('dense cache vectors are not L2-normalized')
    return DenseAliasIndex(entries, vecs, manifest)


def candidate_passage(candidate:dict)->str:
    alias=candidate.get('matched_alias') or ''
    name=candidate.get('canonical_name') or ''
    return f"{name} | {alias}" if alias and alias != name else name

def rerank_candidates(query:str, candidates:list[dict], reranker, batch_size=16):
    pairs=[(query, candidate_passage(c)) for c in candidates]
    scores=[]
    for start in range(0, len(pairs), max(1,batch_size)):
        scores.extend(reranker.score(pairs[start:start+max(1,batch_size)]))
    out=[]
    for c,s in zip(candidates,scores):
        row=dict(c); row['reranker_score']=float(s); row['pre_rerank_rank']=row.get('rank'); out.append(row)
    out=sorted(out, key=lambda x:(-x['reranker_score'], x.get('code','')))
    for i,row in enumerate(out,1): row['final_rank']=i
    if {c.get('code') for c in candidates}!={c.get('code') for c in out}: raise ValueError('reranking changed candidate membership')
    return out

def tune_candidate_strategy(dev_examples, dense_ranker, hybrid_ranker):
    strategies={
        'dense_top20': lambda ex: dense_ranker(ex)[:20],
        'hybrid_top20': lambda ex: hybrid_ranker(ex)[:20],
        'union_dense20_hybrid10': lambda ex: list(dict.fromkeys(dense_ranker(ex)[:20] + hybrid_ranker(ex)[:10])),
    }
    best=None
    for name,fn in strategies.items():
        recalls=[]; counts=[]
        for ex in dev_examples:
            codes=fn(ex); recalls.append(ex['positive_code'] in codes); counts.append(len(codes))
        recall=sum(recalls)/(len(recalls) or 1); avg=sum(counts)/(len(counts) or 1); key=(recall,-avg,name)
        if best is None or key>best[0]: best=(key,{'strategy':name,'candidate_recall':recall,'avg_candidate_count':avg})
    return best[1]

def tune_blend(dev_examples, retrieval_rows, reranked_rows, weights=(0.0,0.25,0.5,0.75,1.0)):
    best=None
    for w in weights:
        ranks=[]
        for ex in dev_examples:
            rr=reranked_rows(ex); by_code={r['code']:r for r in rr}; rows=[]
            for r in retrieval_rows(ex):
                code=r['code']; rows.append({**r, 'blend_score': (1-w)*float(r.get('score',0.0)) + w*float(by_code.get(code,{}).get('reranker_score',0.0))})
            codes=[r['code'] for r in sorted(rows,key=lambda x:(-x['blend_score'],x['code']))]
            ranks.append(codes.index(ex['positive_code'])+1 if ex['positive_code'] in codes else None)
        m=metrics_from_ranks(ranks); key=(m['mrr'],m['recall@1'],w)
        if best is None or key>best[0]: best=(key,{'blend_weight':w,'dev_metrics':m})
    return best[1]

def rrf_fuse(rankings:list[tuple[float,list[str]]], k=60, top_k=10):
    scores={}
    for weight,codes in rankings:
        for rank,code in enumerate(codes,1): scores[code]=scores.get(code,0.0)+weight/(k+rank)
    return [c for c,_ in sorted(scores.items(), key=lambda x:(-x[1],x[0]))[:top_k]]

def metrics_from_ranks(ranks):
    n=len(ranks) or 1
    return {'recall@1':sum(1 for r in ranks if r and r<=1)/n,'recall@5':sum(1 for r in ranks if r and r<=5)/n,'recall@10':sum(1 for r in ranks if r and r<=10)/n,'recall@20':sum(1 for r in ranks if r and r<=20)/n,'mrr':sum(1/r for r in ranks if r)/n}

def tune_rrf(dev_examples, bm25_ranker, dense_ranker, weights=(0.0,0.5,1.0,2.0), ks=(10,60)):
    best=None
    for wb in weights:
      for wd in weights:
       for k in ks:
        if wb == 0.0 and wd == 0.0: continue
        ranks=[]
        for ex in dev_examples:
            bm=bm25_ranker(ex); de=dense_ranker(ex); fused=rrf_fuse([(wb,bm),(wd,de)], k=k, top_k=20)
            ranks.append(fused.index(ex['positive_code'])+1 if ex['positive_code'] in fused else None)
        m=metrics_from_ranks(ranks); key=(m['recall@10'],m['mrr'],m['recall@1'], -wb, -wd, -k)
        if best is None or key>best[0]: best=(key,{'bm25_weight':wb,'dense_weight':wd,'rrf_k':k,'dev_metrics':m})
    return best[1]
