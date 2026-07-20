# Offline Submission Runner

Phase 9 freezes the verified champion pipeline into a local bundle that must run without network access.

## Bundle layout

```text
bundle/
  champion_manifest.json
  SHA256SUMS.txt
  artifact_inventory.json
  models/
    ner/
    assertion/
    bge-m3/
  kb/rxnorm/
  indexes/rxnorm_bge_m3/
  artifacts/
```

The bundle must contain only local model snapshots, tokenizer/config files, assertion `thresholds.json`, official verified RxNorm CPC files, and the validated dense index. Do not include Hugging Face cache folders, optimizer state, raw downloads, logs, or competition outputs.

## Build bundle

```bash
python scripts/build_submission_bundle.py \
  --code-root . \
  --ner-model /path/to/ner_xlmr_large_phase5_v2_1 \
  --assertion-model /path/to/assertion_xlmr_base_v1 \
  --rxnorm-kb /path/to/rxnorm_kb \
  --rxnorm-dense-index /path/to/rxnorm_dense_index \
  --bge-model /path/to/bge-m3-local-snapshot \
  --output /path/to/submission_bundle
```

Use `--dry-run` first to check file counts/sizes without copying large files.

## Preflight

```bash
python scripts/submission_preflight.py \
  --bundle-root /path/to/submission_bundle \
  --manifest /path/to/submission_bundle/champion_manifest.json
```

Preflight validates required files, checksums/inventory, assertion thresholds, verified RxNorm records, candidate universe, dense index manifest, local BGE snapshot, device/dtype, and offline readiness.

## Offline sanity run

```bash
python submission/run.py tests/fixtures/pipeline/input /tmp/submission_output \
  --bundle-root /path/to/submission_bundle \
  --expected-count 2
```

The runner sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `TOKENIZERS_PARALLELISM=false`, uses low-VRAM staged execution, dense-only RxNorm, reranker disabled, and `include_unverified=false`.

## Full input run

```bash
python submission/run.py /path/to/private/input /path/to/output \
  --bundle-root /path/to/submission_bundle \
  --resume \
  --expected-count 100
```

`--resume` skips only already-valid JSON outputs. Corrupt/missing outputs are rerun. One failed document returns a non-zero exit code and does not write a partial JSON.

## Validate

```bash
python scripts/validate_submission.py \
  --input-dir /path/to/private/input \
  --output-dir /path/to/output \
  --expected-count 100 \
  --rxnorm-kb /path/to/submission_bundle/kb/rxnorm
```

The validator enforces exact offsets, deterministic sorting, allowed keys only (`text`, `type`, `position`, `assertions`, `candidates`), assertion order, candidate string lists, verified RxCUIs, and no `relations`, scores, metadata, or internal provenance.

## Package

```bash
python scripts/package_submission.py \
  --input-dir /path/to/private/input \
  --output-dir /path/to/output \
  --rxnorm-kb /path/to/bundle/kb/rxnorm \
  --zip-path /path/to/output.zip \
  --expected-count 100
```

The ZIP contains JSON files at root only with deterministic ordering/timestamps and a package report with SHA256.

## Recovery / resume

If a run is interrupted, rerun `submission/run.py` with `--resume`. The runner validates existing outputs before skipping them.

## Expected resources

The verified Colab run used CUDA FP16 with approximately 1.15 GB peak VRAM for the champion pipeline. Ensure enough disk space for local NER, assertion, BGE-M3, official RxNorm KB, and dense index artifacts.

## Known blocker

Official ICD-10 KB is still missing. Diagnosis `candidates=[]` is expected until official verified ICD-10 data is supplied. The Hugging Face auxiliary 53-code ICD source is **never** allowed for production candidates.
