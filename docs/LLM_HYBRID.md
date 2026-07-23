LLM hybrid challenger

This challenger changes only entity extraction and assertion prediction. Characteroffsets, RxNorm/ICD-10 linking, schema validation, and submission packaging remaindeterministic pipeline stages. The LLM is never allowed to generate candidate codes.

Safety boundary

Do not train on Turn 2 or any official evaluation input.

Do not use the synthetic batch as the champion without an A/B improvement on held-out real gold.

Keep the pre-synthetic checkpoint as the baseline until this challenger wins on the same dev split.

Checkpoints and adapters remain ignored by Git. Only source code is committed.

Build SFT records

The input and annotation directories must have matching stems (1.txt and1.json). Every annotation must satisfy the exact substring invariant.

python -m scripts.build_llm_sft_dataset \
  --input-dir /path/to/released_gold/input \
  --annotations-dir /path/to/released_gold/output \
  --output-dir data/processed/llm_sft \
  --source-role released_gold \
  --dev-ratio 0.2

Candidate codes are deliberately removed from the SFT target. The output containsdocument-disjoint train.jsonl, dev.jsonl, and a split manifest.

QLoRA fine-tuning

Install the optional environment and train a local model that complies with thecompetition parameter limit:

python -m pip install -r requirements-llm.txt
python -m scripts.train_llm_qlora \
  --model /path/to/qwen-7b-or-8b \
  --train data/processed/llm_sft/train.jsonl \
  --dev data/processed/llm_sft/dev.jsonl \
  --output-dir checkpoints/llm_hybrid_adapter

The trainer masks system/user tokens, trains only the assistant JSON response, andsaves a PEFT adapter. Run it on CUDA; it intentionally refuses CPU training.

Run the challenger

Set local artifact paths and use the challenger config:

export NER_MODEL_PATH=/path/to/base-model
export LLM_ADAPTER_PATH=checkpoints/llm_hybrid_adapter
export RXNORM_KB_DIR=/path/to/rxnorm-kb
export RXNORM_DENSE_INDEX_DIR=/path/to/rxnorm-index
export RXNORM_BGE_MODEL_PATH=/path/to/bge-m3
export ICD10_KB_DIR=/path/to/icd10-kb
python -m scripts.run_pipeline_v2 input_dir output_llm \
  --config configs/pipeline.llm_hybrid.challenger.json

Compare this output with the pre-synthetic baseline on the exact same held-out realgold. Promote it only if WER, assertion Jaccard, candidate Jaccard, and the finalscore do not regress. The runtime report must show ner:llm_hybrid, exact offsets,no unverified candidates, and no production blockers.