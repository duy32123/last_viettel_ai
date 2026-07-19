import json, zipfile, pytest
from pathlib import Path
from src.data.import_rxnorm import import_rxnorm_prescribable, parse_rxnconso_line, RXNCONSO_FIELDS
from src.data.kb_schema import KBRecord, write_jsonl
from src.linking.normalization import normalize_mention
from src.linking.pipeline import link_rows
from src.linking.retrieval import LexicalIndex
from src.linking.dense import MockReranker, rerank_candidates

def conso(rxcui, lat='ENG', sab='RXNORM', tty='IN', name='metformin', suppress='N', pref='Y'):
    vals={'RXCUI':rxcui,'LAT':lat,'TS':'','LUI':'','STT':'','SUI':'','ISPREF':pref,'RXAUI':f'A{rxcui}{tty}','SAUI':'','SCUI':'','SDUI':'','SAB':sab,'TTY':tty,'CODE':rxcui,'STR':name,'SRL':'','SUPPRESS':suppress,'CVF':'4096'}
    return '|'.join(vals[f] for f in RXNCONSO_FIELDS)+'|'

def make_zip(tmp_path, lines, rel_lines=None):
    path=tmp_path/'rxnorm.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('rrf/RXNCONSO.RRF','\n'.join(lines)+'\n')
        if rel_lines is not None: z.writestr('rrf/RXNREL.RRF','\n'.join(rel_lines)+'\n')
    return path

def test_rxnconso_field_parsing_and_malformed_rows():
    row=parse_rxnconso_line(conso('6809', tty='IN', name='metformin'))
    assert row['RXCUI']=='6809' and row['SAB']=='RXNORM' and row['TTY']=='IN'
    with pytest.raises(ValueError): parse_rxnconso_line('too|short|')

def test_streaming_zip_filters_sab_lat_suppress_and_verified_provenance(tmp_path):
    z=make_zip(tmp_path,[conso('6809', name='metformin'), conso('6809', tty='SCD', name='metformin 500 MG Oral Tablet', pref='N'), conso('1', sab='MTHSPL', name='bad'), conso('2', suppress='Y', name='suppressed'), conso('3', lat='SPA', name='spanish')])
    records, report=import_rxnorm_prescribable(z, version='2026-07', source_url='url')
    assert report['accepted_rows']==2 and report['concept_count']==1
    rec=records[0]
    assert rec.code=='6809' and rec.terminology=='RxNorm' and rec.verified is True
    assert rec.metadata['official_kb'] is True and rec.metadata['license_required'] is False
    assert rec.metadata['zip_checksum'] and rec.metadata['alias_provenance'][0]['SAB']=='RXNORM'
    assert all(a!='bad' for a in rec.aliases)

def test_deterministic_canonical_and_tty_specificity(tmp_path):
    lines=[conso('111', tty='SCD', name='Drug 10 MG Oral Tablet', pref='Y'), conso('111', tty='IN', name='drug', pref='N'), conso('111', tty='BN', name='BrandDrug', pref='N')]
    z1=make_zip(tmp_path, lines); z2=make_zip(tmp_path, list(reversed(lines)))
    r1,_=import_rxnorm_prescribable(z1, version='v'); r2,_=import_rxnorm_prescribable(z2, version='v')
    assert r1[0].canonical_name == r2[0].canonical_name == 'drug'
    assert r1[0].metadata['TTY']=='IN' and r1[0].metadata['specificity']=='ingredient'

def test_medication_normalization_preserves_offsets_and_extracts_context():
    text='Bệnh nhân dùng thuốc Metformin 500 mg đường uống.'
    start=text.index('Metformin'); end=start+len('Metformin')
    n=normalize_mention('thuốc Metformin 500 mg đường uống','THUỐC')
    assert n['base']=='metformin' and n['features']['strength']=='500 mg' and n['features']['route']=='đường uống'
    assert text[start:end]=='Metformin'

def test_generic_brand_and_dose_context_routing_candidate_schema(tmp_path):
    recs=[KBRecord('1','metformin',['metformin 500 MG Oral Tablet'],'RxNorm','v','RxNorm',True,{'TTY':'IN','specificity':'ingredient'},'drug','en'), KBRecord('2','Glucophage',[],'RxNorm','v','RxNorm',True,{'TTY':'BN','specificity':'brand'},'drug','en'), KBRecord('3','metformin 500 MG Oral Tablet',['metformin tablet'],'RxNorm','v','RxNorm',True,{'TTY':'SCD','specificity':'product'},'drug','en')]
    idx=LexicalIndex(recs)
    gen=idx.search('metformin','THUỐC',top_k=3)
    assert gen[0].code=='1'
    dose=idx.search('metformin 500 mg đường uống','THUỐC',top_k=3)
    assert any(getattr(c,'tty',None)=='SCD' for c in dose)
    brand=idx.search('đang sử dụng Glucophage mỗi ngày','THUỐC',top_k=3)
    assert brand and brand[0].code=='2'
    d=brand[0].to_dict(); assert {'code','terminology','canonical_name','score','rank','retrieval_method','matched_alias','verified','source','version','tty'} <= set(d)

def test_unresolved_and_production_verified_only_behavior():
    unverified=[KBRecord('9','fakeDrug',[],'RxNorm','v','seed',False,{'TTY':'IN'},'drug','en')]
    assert LexicalIndex(unverified).search('fakeDrug','THUỐC') == []
    assert LexicalIndex(unverified, include_unverified=True).search('fakeDrug','THUỐC')
    assert LexicalIndex([], include_unverified=True).search('unknown','THUỐC') == []

def test_split_by_rxcui_no_leakage_and_link_rows_offsets(tmp_path):
    from scripts.build_rxnorm_pilot import split_code, context_for
    rec=KBRecord('6809','metformin',[],'RxNorm','v','RxNorm',True,{'TTY':'IN'},'drug','en')
    examples=context_for(rec); assert len({split_code(rec.code)})==1 and all(e['positive_code']=='6809' for e in examples)
    idx=LexicalIndex([rec])
    row={'text':'Uống metformin mỗi ngày.','entities':[{'text':'metformin','type':'THUỐC','position':[5,14]}]}
    out=link_rows([row], idx, top_k=1)[0]
    assert out['entities'][0]['text']=='metformin' and out['entities'][0]['candidates'][0]['code']=='6809'

def test_reranker_candidate_membership_gate():
    cands=[{'code':'1','canonical_name':'metformin','matched_alias':'metformin','score':1.0,'rank':1},{'code':'2','canonical_name':'Glucophage','matched_alias':'Glucophage','score':0.5,'rank':2}]
    out=rerank_candidates('Glucophage', cands, MockReranker({('Glucophage','Glucophage'):2.0}), batch_size=1)
    assert {r['code'] for r in out} == {'1','2'} and out[0]['code']=='2'
