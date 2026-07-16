# External data sources

PhoNER_COVID19: official repository is recorded in `configs/datasets.yaml`. It is Vietnamese COVID/news medical NER; labels unrelated to the target schema map to `IGNORE`. Verify upstream license before downloading.

VietMed-NER in MultiMed and ViMQ are registered as potential sources with access/license notes. This PR does not download them or claim they cover assertion, linking, or relation training. Fixtures only validate adapter contracts.
