from __future__ import annotations

import json

import pytest

from scripts.build_llm_sft_dataset import build_dataset
from src.models.llm_hybrid.extractor import LLMHybridExtractor, align_predictions, parse_json_response


def test_parse_fenced_and_salvage_truncated_json():
    fenced='```json\n[{"text":"ho","type":"TRIỆU_CHỨNG"}]\n```'
    assert parse_json_response(fenced)[0]['text'] == 'ho'
    truncated='[{"text":"ho","type":"TRIỆU_CHỨNG"},{"text":"sốt"'
    assert parse_json_response(truncated) == [{'text':'ho','type':'TRIỆU_CHỨNG'}]


def test_exact_alignment_uses_hint_for_repeated_mentions_and_drops_hallucination():
    text='Khó thở lúc vào viện. Hiện không còn khó thở.'
    second=text.rfind('khó thở')
    rows=[
        {'text':'Khó thở','type':'TRIỆU_CHỨNG','start':0,'end':8,'assertions':[]},
        {'text':'khó thở','type':'TRIỆU_CHỨNG','start':second+2,'end':999,'assertions':['isNegated']},
        {'text':'đau ngực','type':'TRIỆU_CHỨNG'},
        {'text':'N/A','type':'TÊN_XÉT_NGHIỆM'},
    ]
    aligned,dropped=align_predictions(text,rows)
    assert [row['start'] for row in aligned] == [0,second]
    assert aligned[1]['assertions'] == ['isNegated']
    assert all(text[row['start']:row['end']] == row['text'] for row in aligned)
    assert {item['reason'] for item in dropped} == {'no_exact_substring','garbage_surface'}


def test_chunk_merge_removes_duplicate_and_nested_partial_spans():
    text='Bệnh nhân dùng metoprolol.'
    def generator(_messages):
        return json.dumps([
            {'text':'meto','type':'THUỐC','start':15,'end':19,'score':0.7},
            {'text':'metoprolol','type':'THUỐC','start':15,'end':25,'score':0.9},
        ],ensure_ascii=False)
    rows=LLMHybridExtractor(generator,max_chunk_chars=100).predict(text)
    assert [(row['text'],row['start'],row['end']) for row in rows] == [('metoprolol',15,25)]


def test_pipeline_can_use_llm_for_entities_and_assertions_without_assertion_checkpoint(monkeypatch):
    import src.pipeline.end_to_end as e2e
    cfg=e2e.load_config('configs/pipeline.end_to_end.smoke.yaml')
    cfg['production']=False
    cfg['ner']={'backend':'llm_hybrid','model_path':'unused','mock':False}
    cfg['assertion']={'source':'ner','mock':False}
    class FakeExtractor:
        generation_calls=0; chunk_count=0; dropped_rows=0; dropped_overlaps=0
        def predict(self,text):
            self.generation_calls += 1; self.chunk_count += 1
            start=text.index('đau ngực')
            return [{'text':'đau ngực','type':'TRIỆU_CHỨNG','start':start,'end':start+len('đau ngực'),'score':1.0,'assertions':['isNegated']}]
    fake=FakeExtractor()
    monkeypatch.setattr(e2e.LLMHybridExtractor,'from_config',lambda cfg: fake)
    pipe=e2e.EndToEndPipeline(cfg)
    out=pipe.infer_document('Bệnh nhân không đau ngực.')
    assert out == [{'text':'đau ngực','type':'TRIỆU_CHỨNG','position':[16,24],'assertions':['isNegated']}]
    report=pipe.serializable_report()
    assert report['executed_backends'][:2] == ['ner:llm_hybrid','assertion:llm_hybrid']
    assert 'assertion' not in report['model_metadata']


def test_sft_builder_preserves_offsets_assertions_and_excludes_candidates(tmp_path):
    inp=tmp_path/'input'; ann=tmp_path/'output'; inp.mkdir(); ann.mkdir()
    text='Không ho. Dùng metformin.'
    (inp/'1.txt').write_text(text,encoding='utf-8')
    rows=[
        {'text':'ho','type':'TRIỆU_CHỨNG','position':[6,8],'assertions':['isNegated']},
        {'text':'metformin','type':'THUỐC','position':[15,24],'candidates':['6809']},
    ]
    (ann/'1.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
    record=build_dataset(inp,ann)[0]
    target=json.loads(record['messages'][-1]['content'])
    assert target[0] == {'text':'ho','type':'TRIỆU_CHỨNG','start':6,'end':8,'assertions':['isNegated']}
    assert 'candidates' not in json.dumps(target)
    assert target[1]['assertions'] == []


def test_sft_builder_rejects_incorrect_ground_truth_offset(tmp_path):
    inp=tmp_path/'input'; ann=tmp_path/'output'; inp.mkdir(); ann.mkdir()
    (inp/'1.txt').write_text('ho',encoding='utf-8')
    (ann/'1.json').write_text(json.dumps([{'text':'ho','type':'TRIỆU_CHỨNG','position':[0,1]}]),encoding='utf-8')
    with pytest.raises(ValueError,match='exact offset invariant'):
        build_dataset(inp,ann)