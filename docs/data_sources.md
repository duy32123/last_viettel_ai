# External data sources

PhoNER_COVID19 is configured with a native BIO/CoNLL adapter. Its broad `SYMPTOM_AND_DISEASE` label is mapped to `UNMAPPED` in `configs/datasets.yaml` until a reviewed disease-vs-symptom split strategy exists; unrelated labels map to `IGNORE`.

VietMed-NER (MultiMed) and ViMQ are registry-only entries with official URLs and license/access notes. They are not claimed as supported adapters, and the pipeline skips them unless user-provided local data plus an adapter are configured.

The default production dataset config skips missing official/user-provided files. Test-only fixtures live under `tests/fixtures` and are wired through dedicated test config files instead of the production data-source config.
