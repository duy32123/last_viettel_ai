# Baseline audit

- Dictionary coverage is low and candidate codes in `dicts.py` are explicitly unverified.
- `DRUG_DICT` contains suspicious/conflicting entries and manual corrections, so it must not be treated as RxNorm truth.
- Entity regexes lack full word-boundary protection and can match substrings inside longer words.
- Assertion cues are line/section heuristics and may scope too broadly.
- The lab regex can consume extra text as the test name before a value.
- Occupied spans depend on manual priority order: drug, diagnosis, lab, symptom.
- There is no trained model, true entity linking, relation extraction, or confidence calibration.
- Output keys are not uniform: lab entities omit `assertions` and `candidates` while drug/diagnosis/symptom include subsets.
