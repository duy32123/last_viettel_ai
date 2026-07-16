import json
import pytest

pytestmark = pytest.mark.integration


def test_hf_tiny_token_classification_smoke(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import BertConfig, BertForTokenClassification, Trainer, TrainingArguments, default_data_collator
    from src.models.ner.labels import LABEL2ID, ID2LABEL
    from src.models.ner.metrics import evaluate_spans
    from src.models.ner.inference import merge_chunk_predictions

    config=BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, num_labels=len(LABEL2ID), id2label={str(k):v for k,v in ID2LABEL.items()}, label2id=LABEL2ID)
    model=BertForTokenClassification(config)
    features=[{"input_ids":[1,2,3,4],"attention_mask":[1,1,1,1],"labels":[-100,LABEL2ID["B-TRIỆU_CHỨNG"],LABEL2ID["L-TRIỆU_CHỨNG"],-100]}]
    args=TrainingArguments(output_dir=str(tmp_path/"ckpt"), max_steps=1, per_device_train_batch_size=1, save_steps=1, report_to=[])
    Trainer(model=model, args=args, train_dataset=features, data_collator=default_data_collator).train()
    model.save_pretrained(tmp_path/"saved")
    reloaded=BertForTokenClassification.from_pretrained(tmp_path/"saved")
    assert reloaded.config.id2label
    pred=merge_chunk_predictions([[{"start":0,"end":3,"type":"TRIỆU_CHỨNG","score":0.9}]])
    metrics=evaluate_spans([{"id":"d","text":"sốt","entities":[{"start":0,"end":3,"type":"TRIỆU_CHỨNG","text":"sốt"}]}],[{"id":"d","text":"sốt","entities":pred}])
    assert metrics["strict_micro"]["f1"] == 1.0
