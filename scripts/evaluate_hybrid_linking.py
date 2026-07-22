from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.import_hf_icd_aux import assert_no_query_kb_leakage
from src.data.kb_schema import KBRecord, read_jsonl
from src.linking.dense import (
    BGEM3Backend,
    DenseAliasIndex,
    MockDenseEncoder,
    kb_checksum,
    load_dense_index,
    metrics_from_ranks,
    rrf_fuse,
    tune_rrf,
)
from src.linking.retrieval import LexicalIndex


def _read_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows_from_pilot(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_records(kb_dir: str | Path) -> tuple[list[KBRecord], list[Path]]:
    paths = sorted(Path(kb_dir).glob("*.jsonl"))
    records: list[KBRecord] = []
    for path in paths:
        records.extend(read_jsonl(path))
    return records, paths


def _finite_l2(vec: list[float], *, label: str) -> None:
    if not all(math.isfinite(float(x)) for x in vec):
        raise ValueError(f"{label} vector contains non-finite values")
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if abs(norm - 1.0) > 1e-4:
        raise ValueError(f"{label} vector is not L2-normalized: norm={norm}")


def full_bm25_rank(index: LexicalIndex, mention: str, all_codes: list[str]) -> list[str]:
    """Return a deterministic full code universe for evaluation, including zero-score codes."""
    hits = index.search(mention, "CHẨN_ĐOÁN", top_k=len(all_codes), use_fuzzy=False)
    ranked = [cand.code for cand in hits]
    seen = set(ranked)
    ranked.extend(code for code in sorted(all_codes) if code not in seen)
    return ranked


def _encode_query_vectors(encoder: Any, texts: list[str], *, batch_size: int, expected_dim: int) -> tuple[list[list[float]], dict[str, Any]]:
    vectors: list[list[float]] = []
    started = time.time()
    for start in range(0, len(texts), max(1, batch_size)):
        batch = texts[start : start + max(1, batch_size)]
        vectors.extend(encoder.encode(batch))
    latency = time.time() - started
    for i, vec in enumerate(vectors):
        if len(vec) != expected_dim:
            raise ValueError(f"query/index dimension mismatch: {len(vec)} != {expected_dim}")
        _finite_l2(vec, label=f"query[{i}]")
    return vectors, {
        "query_vector_count": len(vectors),
        "query_encoding_latency_sec": latency,
        "query_encoding_throughput_qps": len(texts) / (latency or 1.0),
    }


def _load_dense_for_eval(
    cfg: dict[str, Any],
    records: list[KBRecord],
    kb_paths: list[Path],
    *,
    mock_dense: bool = False,
    encoder_factory: Callable[..., Any] | None = None,
) -> tuple[DenseAliasIndex, Any, dict[str, Any]]:
    include_unverified = bool(cfg.get("include_unverified", False))
    model_name = cfg.get("model_name", "BAAI/bge-m3")
    model_revision = cfg.get("model_revision", "main")
    max_length = int(cfg.get("max_length", 8192))
    checksum = kb_checksum(kb_paths) if kb_paths else ""
    if mock_dense:
        encoder = MockDenseEncoder()
        manifest = {
            "model_name": model_name,
            "model_revision": model_revision,
            "kb_checksum": checksum,
            "normalization": "l2",
            "include_unverified": include_unverified,
            "max_length": max_length,
            "dimension": encoder.dim,
            "mock_encoder": True,
        }
        index = DenseAliasIndex.build(records, encoder, include_unverified, manifest=manifest)
        cache_validation = "mock_index_rebuilt_explicitly"
    else:
        expected = {
            "model_name": model_name,
            "model_revision": model_revision,
            "kb_checksum": checksum,
            "normalization": "l2",
            "include_unverified": include_unverified,
            "max_length": max_length,
        }
        index = load_dense_index(Path(cfg["dense_index_dir"]), expected)
        factory = encoder_factory or BGEM3Backend
        encoder = factory(
            model_name=model_name,
            batch_size=int(cfg.get("batch_size", 16)),
            max_length=max_length,
            use_fp16=cfg.get("use_fp16"),
            device=cfg.get("device"),
        )
        cache_validation = "valid"
    dim = len(index.vectors[0]) if index.vectors else int(index.manifest.get("dimension", 0) or 0)
    norms = [math.sqrt(sum(float(x) * float(x) for x in vec)) for vec in index.vectors]
    if not mock_dense and model_name == "BAAI/bge-m3" and dim not in (0, 1024):
        raise ValueError(f"unexpected BGE-M3 dimension: {dim}; expected 1024")
    return index, encoder, {
        "model_name": model_name,
        "revision": model_revision,
        "used_mock_encoder": mock_dense,
        "index_dimension": dim,
        "alias_vector_count": len(index.vectors),
        "unique_code_count": len({entry.code for entry in index.entries}),
        "kb_checksum": checksum,
        "cache_validation_result": cache_validation,
        "device": getattr(encoder, "device", None) or ("mock" if mock_dense else "auto"),
        "dtype": "mock" if mock_dense else ("fp16_cuda_if_available_else_fp32"),
        "vector_norm_min": min(norms) if norms else None,
        "vector_norm_max": max(norms) if norms else None,
    }


def _evaluate_ranker(splits: dict[str, list[dict[str, Any]]], ranker: Callable[[dict[str, Any]], list[str]]) -> dict[str, Any]:
    out: dict[str, Any] = {"splits": {}, "per_language": {}}
    all_ranks: list[int | None] = []
    lang_ranks: dict[str, list[int | None]] = defaultdict(list)
    returned_counts: list[int] = []
    for split, examples in splits.items():
        started = time.time()
        ranks: list[int | None] = []
        for ex in examples:
            codes = ranker(ex)
            returned_counts.append(len(codes))
            rank = codes.index(ex["positive_code"]) + 1 if ex["positive_code"] in codes else None
            ranks.append(rank)
            all_ranks.append(rank)
            lang_ranks[ex.get("language", "unknown")].append(rank)
        metrics = metrics_from_ranks(ranks)
        latency = time.time() - started
        metrics.update({"latency_sec": latency, "throughput_qps": len(examples) / (latency or 1.0), "query_count": len(examples)})
        out["splits"][split] = metrics
    out["overall"] = metrics_from_ranks(all_ranks)
    out["per_language"] = {lang: metrics_from_ranks(ranks) for lang, ranks in sorted(lang_ranks.items())}
    if "vi" in out["per_language"]:
        out["vietnamese"] = out["per_language"]["vi"]
    counts = Counter(returned_counts)
    out["returned_candidate_count_distribution"] = {str(k): v for k, v in sorted(counts.items())}
    return out


def _alias_diagnostic(records: list[KBRecord], dense: DenseAliasIndex, encoder: Any, *, batch_size: int, dim: int) -> dict[str, Any]:
    queries: list[tuple[str, str]] = []
    seen_alias: set[str] = set()
    for rec in records:
        for alias in [rec.canonical_name, *rec.aliases]:
            norm = alias.casefold().strip()
            if norm in seen_alias:
                continue
            seen_alias.add(norm)
            queries.append((alias, rec.code))
    vectors, timing = _encode_query_vectors(encoder, [q for q, _ in queries], batch_size=batch_size, expected_dim=dim)
    ranks: list[int | None] = []
    for (_, code), vec in zip(queries, vectors):
        codes = [row["code"] for row in dense.search_vector(vec, top_k=len(records))]
        ranks.append(codes.index(code) + 1 if code in codes else None)
    return {"task": "diagnostic_exact_alias_to_code", "official_evaluation": False, **metrics_from_ranks(ranks), **timing}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/linking.bge_m3_pilot.yaml")
    parser.add_argument("--pilot-examples", default="data/processed/linking_kb_auxiliary/auxiliary_pilot_examples.json")
    parser.add_argument("--mode", choices=["bm25", "bge_dense", "hybrid_rrf"], default=None)
    parser.add_argument("--mock-dense", action="store_true", help="Use explicit test-only mock dense encoder; invalid for readiness.")
    ns = parser.parse_args(argv)
    cfg = _read_config(ns.config)
    records, kb_paths = _load_records(cfg["kb_dir"])
    splits = _rows_from_pilot(ns.pilot_examples)
    leak = assert_no_query_kb_leakage(records, splits)
    include_unverified = bool(cfg.get("include_unverified", False))
    all_codes = sorted({rec.code for rec in records if rec.verified or include_unverified})
    bm25 = LexicalIndex(records, include_unverified=include_unverified)

    mode = ns.mode or cfg.get("mode", "hybrid_rrf")
    batch_size = int(cfg.get("batch_size", 16))
    dense = encoder = None
    preflight: dict[str, Any] | None = None
    dense_vectors_by_id: dict[str, list[float]] = {}

    def bm_rank(ex: dict[str, Any]) -> list[str]:
        return full_bm25_rank(bm25, ex["journal_note"], all_codes)

    if mode in {"bge_dense", "hybrid_rrf"}:
        dense, encoder, preflight = _load_dense_for_eval(cfg, records, kb_paths, mock_dense=ns.mock_dense)
        if preflight["unique_code_count"] != len(all_codes):
            raise ValueError(f"dense candidate universe mismatch: {preflight['unique_code_count']} != {len(all_codes)}")
        texts: list[str] = []
        keys: list[str] = []
        for split, rows in splits.items():
            for row in rows:
                keys.append(f"{split}:{row['id']}")
                texts.append(row["journal_note"])
        vectors, timing = _encode_query_vectors(encoder, texts, batch_size=batch_size, expected_dim=preflight["index_dimension"])
        dense_vectors_by_id = dict(zip(keys, vectors))
        preflight.update(timing)

    def de_rank(ex: dict[str, Any]) -> list[str]:
        if dense is None:
            raise RuntimeError("dense ranker requested without dense index")
        # id may not be globally unique in malformed fixtures; find by split wrapper when possible.
        key = ex.get("_eval_key")
        vec = dense_vectors_by_id[key] if key else None
        rows = dense.search_vector(vec, top_k=len(all_codes))
        ranked = [row["code"] for row in rows]
        ranked.extend(code for code in all_codes if code not in set(ranked))
        return ranked

    keyed_splits: dict[str, list[dict[str, Any]]] = {}
    for split, rows in splits.items():
        keyed_splits[split] = [{**row, "_eval_key": f"{split}:{row['id']}"} for row in rows]

    bm25_eval = _evaluate_ranker(keyed_splits, bm_rank)
    params = None
    if mode == "hybrid_rrf":
        if dense is None:
            raise RuntimeError("hybrid_rrf requires a dense index")
        params = tune_rrf(keyed_splits.get("dev", []), bm_rank, de_rank)
        ranker = lambda ex: rrf_fuse([(params["bm25_weight"], bm_rank(ex)), (params["dense_weight"], de_rank(ex))], params["rrf_k"], top_k=len(all_codes))
    elif mode == "bge_dense":
        ranker = de_rank
    else:
        ranker = bm_rank

    out = {
        "official_evaluation": False,
        "task": "note_to_code_auxiliary",
        "task_description": "document-level auxiliary journal_note-to-ICD retrieval; not true entity mention linking",
        "retrieval_mode": mode,
        "candidate_count": len(all_codes),
        "query_kb_overlap": leak["query_kb_overlap"],
        "mock_encoder": bool(ns.mock_dense),
    }
    if preflight is not None:
        out["dense_preflight"] = preflight
    if params is not None:
        out["selected_on_dev"] = params
    out.update(_evaluate_ranker(keyed_splits, ranker))

    if mode in {"bge_dense", "hybrid_rrf"} and dense is not None and encoder is not None:
        out["diagnostic_exact_alias"] = _alias_diagnostic(records, dense, encoder, batch_size=batch_size, dim=preflight["index_dimension"])

    test_metrics = out["splits"].get("test", {})
    if ns.mock_dense:
        out["readiness"] = "INVALID_MOCK_RUN"
    elif mode == "hybrid_rrf":
        test_mrr_beats_bm25 = test_metrics.get("mrr", 0.0) > bm25_eval["splits"].get("test", {}).get("mrr", 0.0)
        out["readiness"] = "READY_FOR_RERANKER" if test_metrics.get("recall@10", 0.0) >= 0.80 and test_metrics.get("recall@20", 0.0) >= 0.90 and test_mrr_beats_bm25 else "NOT_READY_FOR_RERANKER"
    else:
        out["readiness"] = "NOT_READY_FOR_RERANKER"
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


if __name__ == "__main__":
    main()
