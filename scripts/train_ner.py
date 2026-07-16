from pathlib import Path
import argparse, json, subprocess, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.quality import build_quality_report, assert_data_gate
from src.models.ner.labels import LABEL2ID, ID2LABEL
from src.models.ner.preprocess import read_jsonl, preprocess_records


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config", default="configs/train_ner.yaml"); p.add_argument("--resume-from-checkpoint"); p.add_argument("--dry-run-smoke", action="store_true")
    ns=p.parse_args(); cfg=json.loads(Path(ns.config).read_text(encoding="utf-8"))
    report=build_quality_report([cfg["train_path"], cfg["dev_path"]]); assert_data_gate(report)
    if ns.dry_run_smoke:
        class SmokeTokenizer:
            is_fast=True
            def __call__(self, text, return_offsets_mapping=True, truncation=True, max_length=256, stride=64, return_overflowing_tokens=True, padding=False):
                body=max_length-2; chunks=[]; start=0
                while True:
                    end=min(len(text), start+body)
                    offs=[(0,0)]+[(i,i+1) for i in range(start,end)]+[(0,0)]
                    ids=[0]+[2]*max(0, end-start)+[1]
                    chunks.append({"input_ids":ids,"attention_mask":[1]*len(ids),"offset_mapping":offs,"overflow_to_sample_mapping":0})
                    if end >= len(text): break
                    start=max(end-stride, start+1)
                return {"input_ids":[c["input_ids"] for c in chunks],"attention_mask":[c["attention_mask"] for c in chunks],"offset_mapping":[c["offset_mapping"] for c in chunks],"overflow_to_sample_mapping":[0]*len(chunks)}
        train_features, train_stats=preprocess_records(read_jsonl(cfg["train_path"]), SmokeTokenizer(), int(cfg["max_length"]), int(cfg["stride"]))
        dev_features, dev_stats=preprocess_records(read_jsonl(cfg["dev_path"]), SmokeTokenizer(), int(cfg["max_length"]), int(cfg["stride"]))
        print(json.dumps({"status":"dry_run_smoke_ok","data_quality_report":report,"train_features":len(train_features),"dev_features":len(dev_features),"train_preprocess_stats":train_stats.__dict__,"dev_preprocess_stats":dev_stats.__dict__}, ensure_ascii=False))
        return
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, EarlyStoppingCallback, set_seed, DataCollatorForTokenClassification
    except Exception as e:
        raise RuntimeError("transformers is required for full NER training; install dependencies before running this command") from e
    set_seed(int(cfg.get("seed",13)))
    tok=AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if not getattr(tok, "is_fast", False): raise ValueError("fast tokenizer is required")
    train_features, train_stats=preprocess_records(read_jsonl(cfg["train_path"]), tok, int(cfg["max_length"]), int(cfg["stride"]))
    dev_features, dev_stats=preprocess_records(read_jsonl(cfg["dev_path"]), tok, int(cfg["max_length"]), int(cfg["stride"]))
    model=AutoModelForTokenClassification.from_pretrained(cfg["model_name"], num_labels=len(LABEL2ID), label2id=LABEL2ID, id2label={str(k):v for k,v in ID2LABEL.items()})
    out=Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    args=TrainingArguments(output_dir=str(out), learning_rate=float(cfg["learning_rate"]), num_train_epochs=float(cfg["epochs"]), per_device_train_batch_size=int(cfg["batch_size"]), per_device_eval_batch_size=int(cfg["batch_size"]), gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]), warmup_ratio=float(cfg["warmup_ratio"]), weight_decay=float(cfg["weight_decay"]), fp16=bool(cfg.get("fp16",False)), bf16=bool(cfg.get("bf16",False)), evaluation_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, seed=int(cfg.get("seed",13)))
    collator=DataCollatorForTokenClassification(tok)
    trainer=Trainer(model=model, args=args, train_dataset=train_features, eval_dataset=dev_features, tokenizer=tok, data_collator=collator, callbacks=[EarlyStoppingCallback(early_stopping_patience=int(cfg.get("early_stopping_patience",2)))])
    sha=subprocess.run(["git","rev-parse","HEAD"], text=True, capture_output=True).stdout.strip()
    (out/"training_manifest.json").write_text(json.dumps({"config":cfg,"data_quality_report":report,"train_preprocess_stats":train_stats.__dict__,"dev_preprocess_stats":dev_stats.__dict__,"git_commit":sha}, ensure_ascii=False, indent=2), encoding="utf-8")
    trainer.train(resume_from_checkpoint=ns.resume_from_checkpoint or cfg.get("resume_from_checkpoint"))
    trainer.save_model(str(out/"best")); tok.save_pretrained(str(out/"best"))

if __name__ == "__main__": main()
