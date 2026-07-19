# External data sources

PhoNER_COVID19 is configured with a native BIO/CoNLL adapter. Its broad `SYMPTOM_AND_DISEASE` label is mapped to `UNMAPPED` in `configs/datasets.yaml` until a reviewed disease-vs-symptom split strategy exists; unrelated labels map to `IGNORE`.

VietMed-NER (MultiMed) and ViMQ are registry-only entries with official URLs and license/access notes. They are not claimed as supported adapters, and the pipeline skips them unless user-provided local data plus an adapter are configured.

The default production dataset config skips missing official/user-provided files. Test-only fixtures live under `tests/fixtures` and are wired through dedicated test config files instead of the production data-source config.

## Auxiliary ICD-10 pilot source: birgermoell/icd10-clinical-notes

Phase 7A includes an optional Hugging Face adapter for `birgermoell/icd10-clinical-notes` via `datasets.load_dataset("birgermoell/icd10-clinical-notes")`. This source is treated strictly as `source_kind="auxiliary_hf"`, `official_kb=false`, `synthetic_notes=true`, `license="CC-BY-4.0"`, and `verified=false`. It is useful for pilot lexical linking experiments and Vietnamese alias extraction when `language="vi"`, but it is not an official ICD-10 KB, is not claimed to provide full ICD-10 coverage, and must not unblock production gates that require official local ICD-10 data or WHO ICD credentials.
