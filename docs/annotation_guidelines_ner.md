# Gold-dev NER annotation guideline

## Entity types

Use only these target labels:

- `TRIỆU_CHỨNG`: patient signs/symptoms, e.g. fever, cough, fatigue, pain.
- `CHẨN_ĐOÁN`: disease/diagnosis names only when the mention is a condition, not a symptom-only phrase.
- `THUỐC`: drug/medicine/vaccine names. Include dosage/route only when they are part of the medication mention in the source annotation.
- `TÊN_XÉT_NGHIỆM`: names of tests, measurements, or lab panels, e.g. glucose, WBC.
- `KẾT_QUẢ_XÉT_NGHIỆM`: literal test values/results and units.

## Ambiguous labels

- Combined disease/symptom labels such as `SYMPTOM_AND_DISEASE` must stay `UNMAPPED` until a human reviewer splits them into `TRIỆU_CHỨNG` or `CHẨN_ĐOÁN`.
- `medical_procedure`/`MEDICAL_PROCEDURE` stays `UNMAPPED` unless the reviewer can justify a target label; do not automatically map all procedures to lab names.

## Offsets

- Do not normalize, tokenize, strip accents, or rewrite text before recording offsets.
- Every entity must satisfy `record.text[start:end] == entity.text`.
- Use Python character offsets, start inclusive and end exclusive.
- Nested/overlapping entities are not allowed in Phase 5A; choose the most specific span or mark the record rejected with notes.

## Review workflow

- `gold_dev.todo.jsonl` records start with `review_status="pending"`.
- A human reviewer must set `review_status="approved"`, fill `reviewer`, and resolve/reject ambiguous entities.
- Codex must not set `metadata.gold_evaluation=true` manually.
- Only `scripts/promote_gold_dev.py` may write `data/processed/dev.gold.jsonl`, and only from approved records passing offset/type/overlap validation.
