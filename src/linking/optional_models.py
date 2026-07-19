from __future__ import annotations
class LazyBGEEmbedder:
    def __init__(self, model_name='BAAI/bge-m3', enabled=False): self.model_name=model_name; self.enabled=enabled; self.model=None
    def load(self):
        if not self.enabled: raise RuntimeError('BGE disabled; pass CLI flag in production environment')
        raise RuntimeError('BGE loading is intentionally lazy and not available in Codex smoke tests')
    def score(self, query, docs):
        if self.model is None: self.load()
class LazyBGEReranker:
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3', enabled=False): self.model_name=model_name; self.enabled=enabled; self.model=None
    def load(self):
        if not self.enabled: raise RuntimeError('reranker disabled; pass CLI flag in production environment')
        raise RuntimeError('reranker loading is intentionally lazy and not available in Codex smoke tests')
    def rerank(self, query, candidates):
        if self.model is None: self.load()
