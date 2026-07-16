from pathlib import Path
import json
from src.models.ner.labels import labels_to_spans
from src.models.ner.preprocess import decode_feature_spans
from src.models.ner.inference import merge_chunk_predictions
from src.models.ner.labels import LABEL2ID, ID2LABEL
from scripts.train_ner import training_args_kwargs, trainer_kwargs
from src.data.adapters.token_bio import load_token_bio_jsonl, DEFAULT_MAPPINGS


def test_decoder_ignores_special_token_predictions_and_invalid_bilou():
    offsets=[(0,0),(0,3),(4,6),(0,0)]
    labels=["U-TRIỆU_CHỨNG","B-TRIỆU_CHỨNG","I-TRIỆU_CHỨNG","L-TRIỆU_CHỨNG"]
    assert labels_to_spans(offsets, labels, text_length=6) == []
    assert labels_to_spans([(0,0),(0,3),(0,0)], ["O","U-TRIỆU_CHỨNG","O"], 3) == [{"start":0,"end":3,"type":"TRIỆU_CHỨNG"}]
    # Incomplete B without L is dropped rather than emitted as a reversed/empty span.
    assert labels_to_spans([(0,1),(1,2)], ["B-TRIỆU_CHỨNG","I-TRIỆU_CHỨNG"], 2) == []


def test_decode_feature_spans_filters_empty_reversed_out_of_range():
    feature={"text":"abc","offset_mapping":[(0,0),(2,1),(0,3),(0,0)]}
    ids=[LABEL2ID["O"], LABEL2ID["U-TRIỆU_CHỨNG"], LABEL2ID["U-TRIỆU_CHỨNG"], LABEL2ID["U-TRIỆU_CHỨNG"]]
    spans=decode_feature_spans(feature, ids)
    assert spans == [{"start":0,"end":3,"type":"TRIỆU_CHỨNG","text":"abc"}]
    assert all(0 <= s["start"] < s["end"] <= len(feature["text"]) and s["text"] for s in spans)


def test_long_multi_chunk_merge_no_duplicate_or_type_conflict():
    chunks=[
        [{"start":0,"end":5,"type":"TRIỆU_CHỨNG","text":"aaaaa","score":0.6}],
        [{"start":0,"end":5,"type":"CHẨN_ĐOÁN","text":"aaaaa","score":0.4}],
        [{"start":50,"end":55,"type":"THUỐC","text":"bbbbb","score":0.9}],
    ]
    merged=merge_chunk_predictions(chunks)
    assert [(m["start"],m["end"],m["type"]) for m in merged] == [(0,5,"TRIỆU_CHỨNG"),(50,55,"THUỐC")]


def test_training_args_compat_and_report_to_none():
    class EvalStrategyArgs:
        def __init__(self, output_dir, eval_strategy=None, **kwargs): pass
    cfg={"learning_rate":2e-5,"gradient_accumulation_steps":1,"warmup_ratio":0.1,"weight_decay":0.01,"seed":13,"batch_size":1}
    kwargs=training_args_kwargs(EvalStrategyArgs, cfg, Path("out"), "steps")
    assert kwargs["eval_strategy"] == "steps"
    assert "evaluation_strategy" not in kwargs
    assert kwargs["report_to"] == "none"
    class EvaluationStrategyArgs:
        def __init__(self, output_dir, evaluation_strategy=None, **kwargs): pass
    kwargs=training_args_kwargs(EvaluationStrategyArgs, cfg, Path("out"), "epoch")
    assert kwargs["evaluation_strategy"] == "epoch"


def test_trainer_kwargs_processing_class_fallback():
    class NewTrainer:
        def __init__(self, model=None, processing_class=None, **kwargs): pass
    kw=trainer_kwargs(NewTrainer, "m", "a", [], [], "tok", "coll", None, [])
    assert kw["processing_class"] == "tok" and "tokenizer" not in kw
    class OldTrainer:
        def __init__(self, model=None, tokenizer=None, **kwargs): pass
    kw=trainer_kwargs(OldTrainer, "m", "a", [], [], "tok", "coll", None, [])
    assert kw["tokenizer"] == "tok"


def test_native_token_bio_adapter_inventory_offsets_and_unmapped(tmp_path):
    path=tmp_path/"vimq_bio.jsonl"
    path.write_text(json.dumps({"id":"bio1","tokens":["Uống","Panadol","khi","sốt"],"ner_tags":["O","B-DRUG","O","B-SYMPTOM_AND_DISEASE"]}, ensure_ascii=False)+"\n", encoding="utf-8")
    rows, report=load_token_bio_jsonl(path,"ViMQ","train","fixture",DEFAULT_MAPPINGS["vimq"])
    assert rows[0]["text"] == "Uống Panadol khi sốt"
    assert rows[0]["entities"][0]["text"] == "Panadol"
    assert rows[0]["entities"][0]["type"] == "THUỐC"
    assert rows[0]["entities"][1]["type"] == "UNMAPPED"
    assert report["source_labels"] == {"DRUG":1,"SYMPTOM_AND_DISEASE":1}
