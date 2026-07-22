from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path):
    rows=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip(): rows.append(json.loads(line))
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description='QLoRA fine-tuning for the hybrid medical extractor.')
    parser.add_argument('--model',required=True,help='Local path or Hugging Face base model (<=9B per competition rules).')
    parser.add_argument('--train',required=True)
    parser.add_argument('--dev',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--max-length',type=int,default=8192)
    parser.add_argument('--epochs',type=float,default=3.0)
    parser.add_argument('--learning-rate',type=float,default=2e-4)
    parser.add_argument('--gradient-accumulation-steps',type=int,default=16)
    parser.add_argument('--lora-r',type=int,default=16)
    parser.add_argument('--lora-alpha',type=int,default=32)
    parser.add_argument('--seed',type=int,default=20260722)
    ns=parser.parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer, TrainingArguments

    if not torch.cuda.is_available(): raise RuntimeError('QLoRA training requires a CUDA GPU')
    tokenizer=AutoTokenizer.from_pretrained(ns.model,use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(ns.model,quantization_config=quant,device_map='auto')
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
    model=get_peft_model(model,LoraConfig(r=ns.lora_r,lora_alpha=ns.lora_alpha,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))

    def encode(row):
        messages=row['messages']; prompt_messages=messages[:-1]
        full=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        prompt=tokenizer.apply_chat_template(prompt_messages,tokenize=False,add_generation_prompt=True)
        encoded=tokenizer(full,truncation=True,max_length=ns.max_length,add_special_tokens=False)
        prompt_ids=tokenizer(prompt,truncation=True,max_length=ns.max_length,add_special_tokens=False)['input_ids']
        labels=list(encoded['input_ids']); mask_len=min(len(prompt_ids),len(labels)); labels[:mask_len]=[-100]*mask_len
        encoded['labels']=labels
        return encoded

    train=Dataset.from_list(_read_jsonl(Path(ns.train))).map(encode,remove_columns=['id','messages','source_sha256','entity_count'])
    dev=Dataset.from_list(_read_jsonl(Path(ns.dev))).map(encode,remove_columns=['id','messages','source_sha256','entity_count'])
    args=TrainingArguments(output_dir=ns.output_dir,num_train_epochs=ns.epochs,learning_rate=ns.learning_rate,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=ns.gradient_accumulation_steps,gradient_checkpointing=True,bf16=torch.cuda.is_bf16_supported(),fp16=not torch.cuda.is_bf16_supported(),logging_steps=5,eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,seed=ns.seed,report_to='none')
    collator=DataCollatorForSeq2Seq(tokenizer=tokenizer,padding=True,label_pad_token_id=-100,pad_to_multiple_of=8)
    trainer=Trainer(model=model,args=args,train_dataset=train,eval_dataset=dev,data_collator=collator)
    trainer.train(); trainer.save_model(ns.output_dir); tokenizer.save_pretrained(ns.output_dir)
    (Path(ns.output_dir)/'training_metadata.json').write_text(json.dumps(vars(ns),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__': main()