from pathlib import Path
import argparse, json, subprocess, sys, inspect
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.assertion.labels import ASSERTION_LABELS, LABEL2ID, ID2LABEL
from src.models.assertion.metrics import multilabel_metrics, tune_thresholds, labels_from_scores
from src.models.assertion.preprocess import make_examples, register_special_tokens, tokenize_examples, FloatMultilabelCollator

def load_jsonl(path):
    p=Path(path); return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []

def git_sha():
    try: return subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception: return None

def cfg_bool(cfg, key, default=False): return bool(cfg.get(key, default))

def training_args_kwargs(TrainingArguments, cfg):
    interval=cfg.get("eval_strategy", cfg.get("evaluation_strategy", "epoch"))
    kwargs=dict(output_dir=cfg["output_dir"], learning_rate=float(cfg["learning_rate"]), num_train_epochs=float(cfg.get("epochs",1)), per_device_train_batch_size=int(cfg["per_device_train_batch_size"]), per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]), gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]), warmup_ratio=float(cfg["warmup_ratio"]), weight_decay=float(cfg["weight_decay"]), fp16=cfg_bool(cfg,"fp16"), bf16=cfg_bool(cfg,"bf16"), save_strategy=interval, logging_steps=int(cfg.get("logging_steps",50)), save_total_limit=int(cfg.get("save_total_limit",1)), load_best_model_at_end=True, metric_for_best_model=cfg.get("metric_for_best_model","macro_f1"), greater_is_better=True, seed=int(cfg.get("seed",83)), report_to="none")
    params=inspect.signature(TrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = interval
    return kwargs

def trainer_kwargs(Trainer, model, args, train_dataset, eval_dataset, tokenizer, collator, compute_metrics, callbacks):
    kwargs=dict(model=model,args=args,train_dataset=train_dataset,eval_dataset=eval_dataset,data_collator=collator,compute_metrics=compute_metrics,callbacks=callbacks)
    params=inspect.signature(Trainer.__init__).parameters
    kwargs["processing_class" if "processing_class" in params else "tokenizer"] = tokenizer
    return kwargs

def compute_metrics_from_logits(eval_pred):
    import numpy as np
    logits, labels=eval_pred
    probs=1/(1+np.exp(-logits))
    pred=labels_from_scores(probs.tolist(), {l:0.5 for l in ASSERTION_LABELS})
    gold=[[lab for lab,v in zip(ASSERTION_LABELS,row) if int(v)==1] for row in labels.tolist()]
    m=multilabel_metrics(gold,pred)
    flat={"micro_f1":m["micro_f1"],"macro_f1":m["macro_f1"],"subset_accuracy":m["subset_accuracy"],"none_accuracy":m["none_accuracy"] or 0.0}
    for lab,vals in m["per_label"].items(): flat[f"{lab}_f1"]=vals["f1"]
    return flat

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config", default="configs/train_assertion.xlmr_base.yaml"); p.add_argument("--dry-run-smoke", action="store_true"); p.add_argument("--resume-from-checkpoint")
    ns=p.parse_args(); cfg=json.loads(Path(ns.config).read_text(encoding="utf-8"))
    train_records=load_jsonl(cfg["train_path"]); dev_records=load_jsonl(cfg["dev_path"]); test_records=load_jsonl(cfg.get("test_path", "data/processed/assertion/test.jsonl"))
    train=make_examples(train_records); dev=make_examples(dev_records); test=make_examples(test_records)
    if ns.dry_run_smoke:
        print(json.dumps({"train_examples":len(train),"dev_examples":len(dev),"test_examples":len(test),"model_name":cfg["model_name"],"report_to":cfg.get("report_to","none"),"resume_from_checkpoint":ns.resume_from_checkpoint,"dry_run":True}, ensure_ascii=False)); sys.exit(0)
    try:
        import numpy as np, torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
        from datasets import Dataset
    except Exception as e:
        raise RuntimeError("Install requirements-train.txt to train assertion model") from e
    torch.manual_seed(int(cfg.get("seed",83)))
    tokenizer=AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    model=AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=3, problem_type="multi_label_classification", label2id=LABEL2ID, id2label={str(k):v for k,v in ID2LABEL.items()})
    register_special_tokens(tokenizer, model)
    def encode(examples):
        rows=[{"input_text":t, "labels":l} for t,l in zip(examples["input_text"], examples["labels"])]
        return tokenize_examples(rows, tokenizer, max_length=int(cfg["max_length"]), padding=False, truncation=True)
    train_ds=Dataset.from_list(train).map(encode, batched=True, remove_columns=list(train[0].keys()))
    dev_ds=Dataset.from_list(dev).map(encode, batched=True, remove_columns=list(dev[0].keys()))
    args=TrainingArguments(**training_args_kwargs(TrainingArguments,cfg))
    callbacks=[EarlyStoppingCallback(early_stopping_patience=int(cfg.get("early_stopping_patience",2)))]
    trainer=Trainer(**trainer_kwargs(Trainer, model, args, train_ds, dev_ds, tokenizer, FloatMultilabelCollator(tokenizer), compute_metrics_from_logits, callbacks))
    trainer.train(resume_from_checkpoint=ns.resume_from_checkpoint)
    out=Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True); trainer.save_model(str(out)); tokenizer.save_pretrained(str(out))
    dev_logits=trainer.predict(dev_ds).predictions; dev_gold=[ex["labels"] for ex in dev]
    dev_probs=(1/(1+np.exp(-dev_logits))).tolist(); thresholds=tune_thresholds(dev_gold, dev_probs)
    (out/"thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")
    th={l:v["threshold"] for l,v in thresholds.items()}
    dev_pred=labels_from_scores(dev_probs, th); dev_labels=[[lab for lab,v in zip(ASSERTION_LABELS,row) if v] for row in dev_gold]
    dev_metrics=multilabel_metrics(dev_labels, dev_pred); (out/"dev_metrics.json").write_text(json.dumps(dev_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if test:
        test_ds=Dataset.from_list(test).map(encode, batched=True, remove_columns=list(test[0].keys()))
        test_probs=(1/(1+np.exp(-trainer.predict(test_ds).predictions))).tolist(); test_gold=[ex["labels"] for ex in test]
        test_pred=labels_from_scores(test_probs, th); test_labels=[[lab for lab,v in zip(ASSERTION_LABELS,row) if v] for row in test_gold]
        (out/"test_metrics.json").write_text(json.dumps(multilabel_metrics(test_labels, test_pred), ensure_ascii=False, indent=2), encoding="utf-8")
    manifest={"config":cfg,"git_sha":git_sha(),"train_examples":len(train),"dev_examples":len(dev),"test_examples":len(test),"label2id":LABEL2ID,"id2label":ID2LABEL}
    (out/"training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
