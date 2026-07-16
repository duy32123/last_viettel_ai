from pathlib import Path
import argparse, json, subprocess, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.quality import build_quality_report, assert_data_gate, load_gate_config
from src.models.ner.labels import LABEL2ID, ID2LABEL
from src.models.ner.preprocess import read_jsonl, preprocess_records, decode_feature_spans
from src.models.ner.inference import merge_chunk_predictions
from src.models.ner.metrics import evaluate_spans


def _cfg_int(cfg, name, fallback):
    return int(cfg.get(name, fallback))


def _cfg_bool(cfg, name, fallback=False):
    return bool(cfg.get(name, fallback))


def group_gold(features):
    docs={}
    for f in features:
        rid=f["record_id"]
        docs.setdefault(rid, {"id":rid,"text":f["text"],"entities":f.get("gold_entities", [])})
    return list(docs.values())


def build_compute_metrics(eval_features, id2label):
    gold_docs=group_gold(eval_features)
    def compute_metrics(eval_pred):
        preds=eval_pred.predictions.argmax(axis=-1)
        by_record={}
        for feature, pred_ids in zip(eval_features, preds):
            spans=decode_feature_spans(feature, pred_ids.tolist(), id2label)
            by_record.setdefault(feature["record_id"], []).append(spans)
        pred_docs=[]
        text_by_id={f["record_id"]: f["text"] for f in eval_features}
        for rid, chunks in by_record.items():
            pred_docs.append({"id":rid,"text":text_by_id[rid],"entities":merge_chunk_predictions(chunks)})
        metrics=evaluate_spans(gold_docs, pred_docs)
        flat={
            "strict_span_f1": metrics["strict_micro"]["f1"],
            "strict_span_precision": metrics["strict_micro"]["precision"],
            "strict_span_recall": metrics["strict_micro"]["recall"],
        }
        for typ, vals in metrics["per_type"].items():
            flat[f"f1_{typ}"]=vals["f1"]
        return flat
    return compute_metrics


class FeatureDataset:
    def __init__(self, features): self.features=features
    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        f=self.features[idx]
        return {"input_ids":f["input_ids"], "attention_mask":f["attention_mask"], "labels":f["labels"]}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config", default="configs/train_ner.yaml"); p.add_argument("--resume-from-checkpoint"); p.add_argument("--dry-run-smoke", action="store_true")
    ns=p.parse_args(); cfg=json.loads(Path(ns.config).read_text(encoding="utf-8"))
    report=build_quality_report([cfg["train_path"], cfg["dev_path"]])
    gate=load_gate_config(cfg.get("data_gate_config"))
    assert_data_gate(report, gate)
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
        print(json.dumps({"status":"dry_run_smoke_ok","gate_mode":gate.mode,"data_quality_report":report,"train_features":len(train_features),"dev_features":len(dev_features),"train_preprocess_stats":train_stats.__dict__,"dev_preprocess_stats":dev_stats.__dict__}, ensure_ascii=False))
        return
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, EarlyStoppingCallback, set_seed, DataCollatorForTokenClassification
    except Exception as e:
        raise RuntimeError("transformers is required for full NER training; install dependencies before running this command") from e
    set_seed(int(cfg.get("seed",13)))
    if cfg.get("metadata", {}).get("experimental_pilot"):
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Pilot XLM-R-large training requires CUDA; run scripts/gpu_preflight.py and use --dry-run-smoke on CPU-only machines")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError("Pilot XLM-R-large training requires torch with CUDA") from e
    tok=AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if not getattr(tok, "is_fast", False): raise ValueError("fast tokenizer is required")
    train_features, train_stats=preprocess_records(read_jsonl(cfg["train_path"]), tok, int(cfg["max_length"]), int(cfg["stride"]))
    dev_features, dev_stats=preprocess_records(read_jsonl(cfg["dev_path"]), tok, int(cfg["max_length"]), int(cfg["stride"]))
    model=AutoModelForTokenClassification.from_pretrained(cfg["model_name"], num_labels=len(LABEL2ID), label2id=LABEL2ID, id2label={str(k):v for k,v in ID2LABEL.items()})
    if cfg.get("gradient_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    out=Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    interval = "steps" if cfg.get("max_steps") else "epoch"
    args=TrainingArguments(
        output_dir=str(out),
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=float(cfg.get("epochs", 1)),
        max_steps=int(cfg.get("max_steps", -1)),
        per_device_train_batch_size=_cfg_int(cfg, "per_device_train_batch_size", cfg.get("batch_size", 1)),
        per_device_eval_batch_size=_cfg_int(cfg, "per_device_eval_batch_size", cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        weight_decay=float(cfg["weight_decay"]),
        fp16=_cfg_bool(cfg,"fp16",False),
        bf16=_cfg_bool(cfg,"bf16",False),
        evaluation_strategy=interval,
        eval_steps=cfg.get("eval_steps"),
        save_strategy=interval,
        save_steps=cfg.get("save_steps"),
        logging_steps=int(cfg.get("logging_steps", 50)),
        save_total_limit=cfg.get("save_total_limit"),
        load_best_model_at_end=True,
        metric_for_best_model=cfg.get("metric_for_best_model", "strict_span_f1"),
        greater_is_better=True,
        seed=int(cfg.get("seed",13)),
        remove_unused_columns=False,
        gradient_checkpointing=_cfg_bool(cfg,"gradient_checkpointing",False),
    )
    collator=DataCollatorForTokenClassification(tok)
    trainer=Trainer(model=model, args=args, train_dataset=FeatureDataset(train_features), eval_dataset=FeatureDataset(dev_features), tokenizer=tok, data_collator=collator, compute_metrics=build_compute_metrics(dev_features, model.config.id2label), callbacks=[EarlyStoppingCallback(early_stopping_patience=int(cfg.get("early_stopping_patience",2)))])
    sha=subprocess.run(["git","rev-parse","HEAD"], text=True, capture_output=True).stdout.strip()
    (out/"training_manifest.json").write_text(json.dumps({"config":cfg,"gate":gate.__dict__,"data_quality_report":report,"train_preprocess_stats":train_stats.__dict__,"dev_preprocess_stats":dev_stats.__dict__,"git_commit":sha}, ensure_ascii=False, indent=2), encoding="utf-8")
    trainer.train(resume_from_checkpoint=ns.resume_from_checkpoint or cfg.get("resume_from_checkpoint"))
    trainer.save_model(str(out/"best")); tok.save_pretrained(str(out/"best"))

if __name__ == "__main__": main()
