import json
import pytest
from pathlib import Path
from src.data.quality import build_quality_report, assert_data_gate
from src.models.ner.labels import LABEL2ID, ID2LABEL
from src.models.ner.preprocess import preprocess_records, decode_feature_spans, validate_entities
from src.models.ner.metrics import evaluate_spans
from src.models.ner.inference import merge_chunk_predictions
from src.data.synthetic.generator import generate, sample

class CharTokenizer:
    is_fast=True
    def __call__(self, text, return_offsets_mapping=True, truncation=True, max_length=16, stride=4, return_overflowing_tokens=True, padding=False):
        body=max_length-2; chunks=[]; start=0
        while True:
            end=min(len(text), start+body)
            offsets=[(0,0)] + [(i,i+1) for i in range(start,end)] + [(0,0)]
            ids=[0]+[ord(text[i])%100+2 for i in range(start,end)]+[1]
            chunks.append({"input_ids":ids,"attention_mask":[1]*len(ids),"offset_mapping":offsets,"overflow_to_sample_mapping":0})
            if end >= len(text): break
            start=max(end-stride, start+1)
        return {"input_ids":[c["input_ids"] for c in chunks],"attention_mask":[c["attention_mask"] for c in chunks],"offset_mapping":[c["offset_mapping"] for c in chunks],"overflow_to_sample_mapping":[0]*len(chunks)}


def rec(text, ents):
    return {"id":"r1","text":text,"entities":[{"id":f"E{i+1}","start":s,"end":e,"text":text[s:e],"type":t,"assertions":[],"candidates":[]} for i,(s,e,t) in enumerate(ents)],"relations":[],"source":"test","source_split":"train"}


def test_unicode_crlf_bilou_multisubtoken_round_trip():
    text="BN sốt\r\nho khan và glucose 7,2"
    r=rec(text, [(3,6,"TRIỆU_CHỨNG"),(8,15,"TRIỆU_CHỨNG"),(19,26,"TÊN_XÉT_NGHIỆM"),(27,30,"KẾT_QUẢ_XÉT_NGHIỆM")])
    features, stats=preprocess_records([r], CharTokenizer(), max_length=64, stride=8)
    assert stats.dropped_boundary_entities == 0
    spans=decode_feature_spans(features[0], features[0]["labels"])
    assert {(s["start"],s["end"],s["type"],s["text"]) for s in spans} >= {(3,6,"TRIỆU_CHỨNG","sốt"),(8,15,"TRIỆU_CHỨNG","ho khan")}
    labs=[ID2LABEL[i] for i in features[0]["labels"] if i != -100]
    assert any(l.startswith("B-TRIỆU_CHỨNG") for l in labs)
    assert any(l.startswith("L-TRIỆU_CHỨNG") for l in labs)


def test_long_document_chunking_and_boundary_entity_kept():
    text="a"*10 + " glucose " + "b"*40
    start=text.index("glucose"); end=start+7
    features, stats=preprocess_records([rec(text, [(start,end,"TÊN_XÉT_NGHIỆM")])], CharTokenizer(), max_length=16, stride=7)
    assert len(features) > 1
    assert any(any(ID2LABEL[i] == "B-TÊN_XÉT_NGHIỆM" for i in f["labels"] if i != -100) for f in features)


def test_ignore_unmapped_duplicate_and_overlap_policy():
    r={"id":"x","text":"abc def","entities":[{"id":"E1","start":0,"end":3,"text":"abc","type":"IGNORE"},{"id":"E2","start":4,"end":7,"text":"def","type":"UNMAPPED"}],"relations":[]}
    feats, stats=preprocess_records([r], CharTokenizer(), 16, 2)
    assert all(l in {-100, LABEL2ID["O"]} for l in feats[0]["labels"])
    assert stats.ignored_entities == 2
    dup=rec("sốt sốt", [(0,3,"TRIỆU_CHỨNG"),(0,3,"TRIỆU_CHỨNG")])
    _, stats=preprocess_records([dup], CharTokenizer(), 16, 2)
    assert stats.duplicate_entities == 1
    overlap=rec("abcdef", [(0,4,"TRIỆU_CHỨNG"),(2,6,"CHẨN_ĐOÁN")])
    with pytest.raises(ValueError): preprocess_records([overlap], CharTokenizer(), 16, 2)


def test_exact_span_metrics_and_prediction_merge():
    gold=[rec("sốt ho", [(0,3,"TRIỆU_CHỨNG"),(4,6,"TRIỆU_CHỨNG")])]
    pred=[{"id":"r1","text":"sốt ho","entities":[{"start":0,"end":3,"type":"TRIỆU_CHỨNG","text":"sốt"},{"start":4,"end":6,"type":"CHẨN_ĐOÁN","text":"ho"}]}]
    m=evaluate_spans(gold, pred)
    assert m["strict_micro"]["tp"] == 1 and m["false_positives"] == 1 and m["false_negatives"] == 1
    assert m["boundary_only"]["tp"] == 2
    assert m["type_accuracy_on_correct_boundary"] == 0.5
    merged=merge_chunk_predictions([[{"start":0,"end":3,"type":"TRIỆU_CHỨNG","score":0.4}],[{"start":0,"end":3,"type":"TRIỆU_CHỨNG","score":0.9}]])
    assert len(merged) == 1 and merged[0]["score"] == 0.9


def test_no_fake_unverified_candidate_and_quality_gate(tmp_path):
    generate(13,{"train":8,"dev":8,"test":4},tmp_path, None, strict_candidates=False)
    for split in ["train","dev","test"]:
        for line in (tmp_path/f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            r=json.loads(line)
            for e in r["entities"]:
                assert all((c if isinstance(c,str) else c["code"]) != "UNVERIFIED" for c in e.get("candidates",[]))
    report=build_quality_report([tmp_path/"train.jsonl", tmp_path/"dev.jsonl"])
    assert report["candidate_coverage"].get("fake_unverified_code",0) == 0
    assert_data_gate(report)


def test_synthetic_candidate_strict_non_strict_and_verified_value(tmp_path):
    kb=tmp_path/"kb.jsonl"
    kb.write_text(json.dumps({"code":"I10","terminology":"ICD-10","verified":True,"preferred_name":"tăng huyết áp","synonyms":[],"normalized_names":["tăng huyết áp"]}, ensure_ascii=False)+"\n", encoding="utf-8")
    out=tmp_path/"out"
    generate(13,{"train":1},out,{"diagnosis":[str(kb)],"drug":[]}, strict_candidates=False)
    row=json.loads((out/"train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    for e in row["entities"]:
        for c in e.get("candidates",[]):
            assert c["code"] != "UNVERIFIED"
            if c["code"] == "I10": assert c["verified"] is True
    with pytest.raises(ValueError, match="missing KB candidate"):
        sample(1, "history_drug", {"diagnosis":{},"drug":{}}, strict_candidates=True)


def test_chunk_boundary_partial_entity_masked_once_no_contradictory_o():
    text="aaaa glucose bbbb"
    start=text.index("glucose"); end=start+7
    features, stats=preprocess_records([rec(text, [(start,end,"TÊN_XÉT_NGHIỆM")])], CharTokenizer(), max_length=9, stride=2)
    full_chunks=[]; partial_chunks=[]
    for f in features:
        overlap=[i for i,(s,e) in enumerate(f["offset_mapping"]) if s!=e and max(s,start) < min(e,end)]
        if not overlap:
            continue
        labels=[f["labels"][i] for i in overlap]
        if f["offset_mapping"][overlap[0]][0] == start and f["offset_mapping"][overlap[-1]][1] == end:
            full_chunks.append(labels)
        else:
            partial_chunks.append(labels)
            assert all(l == -100 for l in labels)
            assert LABEL2ID["O"] not in labels
    assert full_chunks, "at least one chunk must fully supervise the entity"
    assert partial_chunks, "at least one chunk must mask a partial entity"
    assert any(ID2LABEL[l] == "B-TÊN_XÉT_NGHIỆM" for labels in full_chunks for l in labels if l != -100)
    assert stats.dropped_boundary_entities == 0
    assert stats.partial_entity_chunks >= 1


def test_full_gate_fails_current_synthetic_data_but_smoke_gate_passes(tmp_path):
    generate(13,{"train":8,"dev":8,"test":4},tmp_path, None, strict_candidates=False)
    report=build_quality_report([tmp_path/"train.jsonl", tmp_path/"dev.jsonl", tmp_path/"test.jsonl"])
    from src.data.quality import load_gate_config
    assert_data_gate(report, load_gate_config("configs/data_gate.smoke.yaml"))
    with pytest.raises(ValueError, match="non-synthetic|not enough|gold|official"):
        assert_data_gate(report, load_gate_config("configs/data_gate.full.yaml"))


def test_trainer_compute_metrics_exact_span_integration():
    from scripts.train_ner import build_compute_metrics
    text="sốt"
    feature={"record_id":"r1","text":text,"offset_mapping":[(0,0),(0,1),(1,2),(2,3),(0,0)],"gold_entities":[{"start":0,"end":3,"text":"sốt","type":"TRIỆU_CHỨNG"}]}
    class EvalPred:
        predictions=None
    pred_ids=[0, LABEL2ID["B-TRIỆU_CHỨNG"], LABEL2ID["I-TRIỆU_CHỨNG"], LABEL2ID["L-TRIỆU_CHỨNG"], 0]
    class FakePredictions:
        def __init__(self, ids): self.ids=ids
        def argmax(self, axis=-1): return [FakeIds(self.ids)]
    class FakeIds(list):
        def tolist(self): return list(self)
    ep=EvalPred(); ep.predictions=FakePredictions(pred_ids)
    m=build_compute_metrics([feature], {str(k):v for k,v in ID2LABEL.items()})(ep)
    assert m["strict_span_f1"] == 1.0


def test_merge_conflicting_types_same_boundary_keeps_highest_score():
    merged=merge_chunk_predictions([
        [{"start":0,"end":3,"type":"TRIỆU_CHỨNG","score":0.4}],
        [{"start":0,"end":3,"type":"CHẨN_ĐOÁN","score":0.9}],
    ])
    assert len(merged) == 1
    assert merged[0]["type"] == "CHẨN_ĐOÁN"

def test_predict_with_model_uses_softmax_span_confidence_not_constant_one():
    torch=pytest.importorskip("torch")
    from types import SimpleNamespace
    from src.models.ner.inference import predict_with_model
    class Tok:
        def __call__(self, text, return_offsets_mapping=True, truncation=True, max_length=16, stride=4, return_overflowing_tokens=True, padding=False):
            return {'input_ids':[0,1,2,3,0], 'attention_mask':[1,1,1,1,1], 'offset_mapping':[(0,0),(0,1),(1,2),(2,3),(0,0)]}
    class Model:
        def parameters(self): return iter([torch.zeros(1)])
        def __call__(self, **kwargs):
            logits=torch.full((1,5,len(LABEL2ID)), -5.0)
            logits[0,0,LABEL2ID['O']]=5.0; logits[0,4,LABEL2ID['O']]=5.0
            logits[0,1,LABEL2ID['B-TRIỆU_CHỨNG']]=3.0
            logits[0,2,LABEL2ID['I-TRIỆU_CHỨNG']]=1.0
            logits[0,3,LABEL2ID['L-TRIỆU_CHỨNG']]=2.0
            return SimpleNamespace(logits=logits)
    rows=predict_with_model('sốt', Tok(), Model())
    assert rows and rows[0]['text'] == 'sốt'
    assert 0.0 < rows[0]['score'] < 1.0
