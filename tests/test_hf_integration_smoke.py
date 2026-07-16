import pytest

pytestmark = pytest.mark.integration


class TinyOffsetTokenizer:
    is_fast = True

    def __call__(self, text, return_offsets_mapping=True, truncation=True, max_length=32, stride=8, return_overflowing_tokens=True, padding=False):
        # Character-level tokenizer with BERT-like special tokens. Token ids are
        # deterministic and fit the tiny local vocab used by the test model.
        body = max_length - 2
        chunks = []
        start = 0
        while True:
            end = min(len(text), start + body)
            offsets = [(0, 0)] + [(i, i + 1) for i in range(start, end)] + [(0, 0)]
            ids = [2] + [3 + (ord(text[i]) % 20) for i in range(start, end)] + [3]
            chunks.append({
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "offset_mapping": offsets,
                "overflow_to_sample_mapping": 0,
            })
            if end >= len(text):
                break
            start = max(end - stride, start + 1)
        return {
            "input_ids": [c["input_ids"] for c in chunks],
            "attention_mask": [c["attention_mask"] for c in chunks],
            "offset_mapping": [c["offset_mapping"] for c in chunks],
            "overflow_to_sample_mapping": [0] * len(chunks),
        }


def test_hf_tiny_token_classification_smoke(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import BertConfig, BertForTokenClassification, Trainer, TrainingArguments, default_data_collator
    from src.models.ner.labels import LABEL2ID, ID2LABEL
    from src.models.ner.preprocess import preprocess_records, decode_feature_spans
    from src.models.ner.inference import merge_chunk_predictions

    text = "BN sốt"
    record = {
        "id": "tiny-1",
        "text": text,
        "entities": [{"id": "E1", "start": 3, "end": 6, "text": text[3:6], "type": "TRIỆU_CHỨNG", "assertions": [], "candidates": []}],
        "relations": [],
        "source": "integration_fixture",
        "source_split": "train",
    }
    tokenizer = TinyOffsetTokenizer()
    features, stats = preprocess_records([record], tokenizer, max_length=16, stride=4)
    assert features and stats.dropped_boundary_entities == 0

    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=len(LABEL2ID),
        id2label={str(k): v for k, v in ID2LABEL.items()},
        label2id=LABEL2ID,
    )
    model = BertForTokenClassification(config)
    train_features = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"], "labels": f["labels"]} for f in features]
    args = TrainingArguments(output_dir=str(tmp_path / "ckpt"), max_steps=1, per_device_train_batch_size=1, save_steps=1, report_to=[])
    Trainer(model=model, args=args, train_dataset=train_features, data_collator=default_data_collator).train()

    model.save_pretrained(tmp_path / "saved")
    reloaded = BertForTokenClassification.from_pretrained(tmp_path / "saved")
    reloaded.eval()
    chunk_predictions = []
    with torch.no_grad():
        for feature in features:
            logits = reloaded(
                input_ids=torch.tensor([feature["input_ids"]]),
                attention_mask=torch.tensor([feature["attention_mask"]]),
            ).logits[0]
            pred_ids = logits.argmax(dim=-1).tolist()
            spans = decode_feature_spans(feature, pred_ids, reloaded.config.id2label)
            for span in spans:
                span["text"] = text[span["start"]:span["end"]]
                assert span["text"] == text[span["start"]:span["end"]]
                assert {"start", "end", "type", "text"}.issubset(span)
            chunk_predictions.append(spans)
    merged = merge_chunk_predictions(chunk_predictions)
    for span in merged:
        assert text[span["start"]:span["end"]] == span["text"]
