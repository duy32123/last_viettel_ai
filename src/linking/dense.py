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
    """Direct Transformers backend for BAAI/bge-reranker-v2-m3.

    This avoids the FlagEmbedding AbsReranker tokenizer.prepare_for_model path
    that is incompatible with some current Transformers tokenizers. Models are
    loaded lazily on first score() call, never at import time.
    """
    backend='transformers'
    def __init__(self, model_name=BGE_RERANKER_MODEL, batch_size=8, max_length=512, use_fp16=None, device=None, revision='main'):
        self.model_name=model_name; self.batch_size=batch_size; self.max_length=max_length; self.use_fp16=use_fp16; self.device=device; self.revision=revision
        self.model=None; self.tokenizer=None; self.dtype=None; self.tokenizer_class=None; self.tokenizer_is_fast=None; self.model_class=None; self.transformers_version=None
    def load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import transformers
        import torch
        cuda=bool(torch.cuda.is_available())
        resolved_device=self.device or ('cuda' if cuda else 'cpu')
        fp16=bool(cuda) if self.use_fp16 is None else bool(self.use_fp16 and cuda)
        dtype=torch.float16 if fp16 else torch.float32
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_name, revision=self.revision, use_fast=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(self.model_name, revision=self.revision, torch_dtype=dtype)
        self.model.to(resolved_device); self.model.eval()
        self.device=resolved_device; self.dtype=str(dtype).replace('torch.','')
        self.tokenizer_class=self.tokenizer.__class__.__name__; self.tokenizer_is_fast=bool(getattr(self.tokenizer,'is_fast',False))
        self.model_class=self.model.__class__.__name__; self.transformers_version=getattr(transformers, '__version__', None)
    def score(self, pairs):
        if self.model is None or self.tokenizer is None: self.load()
        import math as _math
        import torch
        rows=list(pairs)
        out=[]
        for start in range(0, len(rows), max(1,self.batch_size)):
            batch=rows[start:start+max(1,self.batch_size)]
            encoded=self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
            encoded={k:v.to(self.device) for k,v in encoded.items()}
            with torch.inference_mode():
                logits=self.model(**encoded).logits.view(-1).float()
            scores=[float(x) for x in logits.detach().cpu().tolist()]
            if len(scores) != len(batch):
                raise ValueError(f'transformers reranker score count mismatch: {len(scores)} != {len(batch)}')
            if not all(_math.isfinite(x) for x in scores):
                raise ValueError('transformers reranker produced non-finite scores')
            out.extend(scores)
        if len(out) != len(rows):
            raise ValueError(f'transformers reranker score count mismatch: {len(out)} != {len(rows)}')
        return out
    def runtime_info(self):
        return {'reranker_backend':self.backend,'transformers_version':self.transformers_version,'tokenizer_class':self.tokenizer_class,'tokenizer_is_fast':self.tokenizer_is_fast,'model_class':self.model_class,'device':self.device,'dtype':self.dtype,'score_validation_passed':self.model is not None}

class FlagEmbeddingLegacyRerankerBackend:
    backend='flagembedding_legacy'
    def __init__(self, model_name=BGE_RERANKER_MODEL, batch_size=16, max_length=8192, use_fp16=None, device=None, revision='main'):
        self.model_name=model_name; self.batch_size=batch_size; self.max_length=max_length; self.use_fp16=use_fp16; self.device=device; self.revision=revision; self.model=None; self.dtype=None
    def load(self):
        from FlagEmbedding import FlagReranker
        if self.use_fp16 is None:
            try:
                import torch; fp16=bool(torch.cuda.is_available())
            except Exception: fp16=False
        else: fp16=bool(self.use_fp16)
        self.dtype='float16' if fp16 else 'float32'
        self.model=FlagReranker(self.model_name, use_fp16=fp16, device=self.device)
    def score(self, pairs):
        if self.model is None: self.load()
        rows=list(pairs); out=[]
        for start in range(0, len(rows), max(1,self.batch_size)):
            batch=rows[start:start+max(1,self.batch_size)]
            scores=self.model.compute_score(batch, batch_size=self.batch_size, max_length=self.max_length)
            if isinstance(scores, (float,int)): scores=[scores]
            if len(scores) != len(batch): raise ValueError(f'flagembedding_legacy reranker score count mismatch: {len(scores)} != {len(batch)}')
            vals=[float(x) for x in scores]
            if not all(math.isfinite(x) for x in vals): raise ValueError('flagembedding_legacy reranker produced non-finite scores')
            out.extend(vals)
        return out
    def runtime_info(self):
        return {'reranker_backend':self.backend,'transformers_version':None,'tokenizer_class':None,'tokenizer_is_fast':None,'model_class':self.model.__class__.__name__ if self.model is not None else None,'device':self.device,'dtype':self.dtype,'score_validation_passed':self.model is not None}

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
    def __init__(self, entries:list[AliasEntry], vectors, manifest:dict[str,Any]|None=None, *, already_normalized:bool=False):
        self.entries=entries; self.manifest=manifest or {}
        if np is not None and hasattr(vectors, 'shape'):
            self.matrix=vectors.astype('float32', copy=False)
            self.vectors=[]
        else:
            self.vectors=[list(map(float, v if already_normalized else l2_normalize(v))) for v in vectors]
            self.matrix = np.asarray(self.vectors, dtype='float32') if np is not None and self.vectors else None
    @classmethod
    def build(cls, records, encoder, include_unverified=False, manifest=None):
        entries=alias_entries(records, include_unverified); vecs=encoder.encode([e.alias for e in entries]) if entries else []
        return cls(entries, vecs, manifest)
    def search(self, query, top_k=10):
        raise RuntimeError('DenseAliasIndex.search requires an explicit encoder or query vector; use search_with_encoder() or search_vector()')
    def search_with_encoder(self, query, encoder, top_k=10):
        return self.search_vector(encoder.encode([query])[0], top_k)
    def _dim(self):
        if self.matrix is not None and getattr(self.matrix, 'ndim', 0)==2: return int(self.matrix.shape[1])
        return len(self.vectors[0]) if self.vectors else 0
    def _scores(self, qvec):
        if self.vectors:
            return [dot(qvec, v) for v in self.vectors]
        if self.matrix is not None:
            return self.matrix @ np.asarray(qvec, dtype='float32')
        return []
    def search_vector(self, qvec, top_k=10):
        dim=self._dim()
        if dim and len(qvec) != dim: raise ValueError(f'query/index dimension mismatch: {len(qvec)} != {dim}')
        norm=math.sqrt(sum(float(x)*float(x) for x in qvec)) or 0.0
        if not all(math.isfinite(float(x)) for x in qvec) or abs(norm-1.0)>1e-4: raise ValueError('query vector must be finite and L2-normalized')
        scores=self._scores(qvec); n=len(self.entries); k=min(max(1, int(top_k)), n)
        alias_k=min(n, max(k, k*5))
        if np is not None and hasattr(scores, 'shape') and n>alias_k:
            idxs=np.argpartition(-scores, alias_k-1)[:alias_k]
            order=sorted((int(i) for i in idxs), key=lambda i:(-float(scores[i]), self.entries[i].code, self.entries[i].alias))
        else:
            order=sorted(range(n), key=lambda i:(-float(scores[i]), self.entries[i].code, self.entries[i].alias))[:alias_k]
        best={}
        for i in order:
            e=self.entries[i]; s=float(scores[i]); cur=best.get(e.code)
            if cur is None or s>cur['score'] or (s==cur['score'] and e.alias<cur['matched_alias']):
                best[e.code]={'code':e.code,'canonical_name':e.canonical_name,'terminology':e.terminology,'score':s,'matched_alias':e.alias,'verified':e.verified,'source':e.source,'version':e.version,'retrieval_method':'bge_dense'}
        rows=sorted(best.values(), key=lambda x:(-x['score'], x['code']))[:k]
        for i,r in enumerate(rows,1): r['rank']=i
        return rows

def kb_checksum(paths):
    h=hashlib.sha256()
    for p in sorted(Path(x) for x in paths): h.update(file_sha256(p).encode())
    return h.hexdigest()


def dense_expected_manifest(cfg:dict, kb_paths, *, dimension:int|None=None, candidate_universe:int|None=None):
    out={'schema_version':1,'model_name':cfg.get('model_name','BAAI/bge-m3'),'model_revision':cfg.get('model_revision','main'),'encoder_backend':cfg.get('encoder_backend','FlagEmbedding.BGEM3FlagModel'),'dtype':cfg.get('dtype','float32'),'normalization':'l2','kb_checksum':kb_checksum(kb_paths) if kb_paths else None,'include_unverified':bool(cfg.get('include_unverified',False)),'max_length':int(cfg.get('max_length',8192)),'batch_size':int(cfg.get('batch_size',16)),'mock_encoder':False}
    if dimension is not None: out['dimension']=int(dimension)
    if candidate_universe is not None: out['candidate_universe']=int(candidate_universe)
    return out

def validate_dense_manifest(manifest:dict, expected:dict):
    required={'schema_version','model_name','model_revision','encoder_backend','dimension','dtype','normalization','kb_checksum','include_unverified','max_length','batch_size','candidate_universe','mock_encoder'}
    missing=sorted(required-set(manifest))
    if missing: raise ValueError(f'dense manifest missing required fields: {missing}')
    for k,v in expected.items():
        if k in manifest and manifest.get(k)!=v: raise ValueError(f'stale dense cache for {k}')
    if manifest.get('normalization')!='l2': raise ValueError('dense cache normalization must be l2')
    return True

def _atomic_write_text(path:Path, text:str):
    tmp=path.with_name(path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)

def save_dense_index(index:DenseAliasIndex, out_dir:Path, manifest:dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest={**manifest}
    manifest.setdefault('schema_version',1); manifest.setdefault('model_revision','main'); manifest.setdefault('encoder_backend','FlagEmbedding.BGEM3FlagModel'); manifest.setdefault('dtype','float32'); manifest.setdefault('normalization','l2'); manifest.setdefault('include_unverified',False); manifest.setdefault('max_length',8192); manifest.setdefault('batch_size',16); manifest.setdefault('candidate_universe',len({e.code for e in index.entries})); manifest.setdefault('mock_encoder',False); manifest.setdefault('dimension', index._dim())

    if np is not None:
        tmp=out_dir/'embeddings.tmp.npy'
        np.save(tmp, index.matrix if index.matrix is not None else np.asarray(index.vectors, dtype='float32'))
        tmp.replace(out_dir/'embeddings.npy')
    else:
        _atomic_write_text(out_dir/'embeddings.json', json.dumps(index.vectors))
    _atomic_write_text(out_dir/'alias_to_code.json', json.dumps([e.__dict__ for e in index.entries], ensure_ascii=False, indent=2))
    # Publish manifest last after vectors and mapping are complete.
    _atomic_write_text(out_dir/'dense_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))

def load_dense_index(out_dir:Path, expected:dict):
    if not (out_dir/'dense_manifest.json').exists(): raise FileNotFoundError(f'dense index missing: {out_dir}')
    manifest=json.loads((out_dir/'dense_manifest.json').read_text())
    validate_dense_manifest(manifest, expected)
    entries=[AliasEntry(**d) for d in json.loads((out_dir/'alias_to_code.json').read_text())]
    dim=int(manifest['dimension'])
    if (out_dir/'embeddings.npy').exists() and np is not None:
        matrix=np.load(out_dir/'embeddings.npy', mmap_mode='r')
        if matrix.ndim!=2 or matrix.shape[1]!=dim: raise ValueError('dense cache vector dimension mismatch')
        for start in range(0, matrix.shape[0], 4096):
            chunk=np.asarray(matrix[start:start+4096], dtype='float32')
            if not np.isfinite(chunk).all(): raise ValueError('dense cache vectors contain non-finite values')
            norms=np.linalg.norm(chunk, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-4): raise ValueError('dense cache vectors are not L2-normalized')
        return DenseAliasIndex(entries, matrix, manifest, already_normalized=True)
    vecs=json.loads((out_dir/'embeddings.json').read_text())
    if vecs and any(len(v)!=dim for v in vecs): raise ValueError('dense cache vector dimension mismatch')
    if vecs and any(abs(math.sqrt(sum(float(x)*float(x) for x in v))-1.0)>1e-4 for v in vecs): raise ValueError('dense cache vectors are not L2-normalized')
    return DenseAliasIndex(entries, vecs, manifest, already_normalized=True)


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
