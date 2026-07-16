# External dataset source audit

This repository does not commit raw upstream datasets. Local downloads must stay under `data/raw/`, which is gitignored.

## VietMed-NER / MultiMed

- URL: https://github.com/leduckhai/MultiMed/tree/master/VietMed-NER
- Observed README: describes VietMed-NER as public code/data/models and says the dataset has 18 entity types, but no explicit redistribution license was confirmed in this environment.
- Policy: local import only; do not commit imported records unless license/terms are clarified.

## ViMQ

- URL: https://github.com/tadeephuy/ViMQ
- Paper/source schema: entity tuples are described as `(start, end, category)`; labels include `SYMPTOM_AND_DISEASE`, `MEDICAL_PROCEDURE`, and `DRUG`/medicine.
- License: no explicit redistribution license was confirmed from the GitHub README in this environment.
- Policy: map `DRUG` to `THUỐC`; keep `SYMPTOM_AND_DISEASE` and `MEDICAL_PROCEDURE` as `UNMAPPED` for review.

## PhoNER_COVID19

- URL: https://github.com/VinAIResearch/PhoNER_COVID19
- Policy: local import only after license/terms check; `SYMPTOM_AND_DISEASE` remains `UNMAPPED` by default.
