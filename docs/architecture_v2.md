# Medical NLP v2 Architecture

Phase 5+ will train separate components. NER uses `FacebookAI/xlm-roberta-large` token classification with BILOU labels and maps token outputs back to original character offsets. Assertion detection is a per-concept XLM-RoBERTa multi-label classifier for `isNegated`, `isFamily`, `isHistorical`, combined with high-precision scoped rules; rule positives override model negatives only inside the same clause/sentence/section.

Candidate retrieval keeps ICD and RxNorm indexes separate. It combines BM25 and `BAAI/bge-m3` dense retrieval by configured Reciprocal Rank Fusion over mention, normalized mention, and local context. Metrics are Recall@1/5/10 and MRR. Reranking uses `BAAI/bge-reranker-v2-m3` on top-N retrieved pairs only.

Relation extraction is an interface with XLM-RoBERTa entity markers, disabled until official labels exist. LLM support (`Qwen/Qwen3-4B-Instruct-2507`) is optional for synthetic data, weak labeling, paraphrase, typo generation, hard-example analysis, and low-confidence fallback. The LLM is not a source of truth for KB codes or offsets, and is disabled for offline scoring by default.

Legacy `extract.py` and `dicts.py` remain a rule-based fallback and weak-label seed only; legacy codes are unverified.
