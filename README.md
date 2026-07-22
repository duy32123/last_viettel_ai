# Viettel AI Race Medical NLP v2 Foundation

Legacy CLI remains: `python run_pipeline.py <input_dir> <output_dir>`.

Phase commands:

```bash
python scripts/export_seed_kb.py
python scripts/build_kb.py --config configs/data_sources.yaml
python scripts/prepare_external_data.py --config configs/datasets.yaml
python scripts/generate_synthetic.py --config configs/synthetic.yaml
pytest -q
```

`data/raw/`, caches, checkpoints, and model weights are ignored. Official data requiring credentials must be placed by the user or configured through environment variables; the pipeline skips missing sources instead of inventing records.
