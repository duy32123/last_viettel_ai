import unicodedata
from .labels import ordered

def _norm(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s.lower()) if unicodedata.category(c)!='Mn')

def rule_assertions(text, entity):
    s,e=entity['position']; left=text[max(0,s-80):s]; before=_norm(left); alltxt=_norm(text); labels=[]; hits=[]
    if ('khong nhung' not in before) and any(c in before for c in ['khong ghi nhan','khong ', 'chua thay','phu nhan','loai tru']): labels.append('isNegated'); hits.append({'rule_id':'neg_scope','cue_span':[0,14]})
    if any(c in before for c in ['me ', 'bo ', 'anh ruot','gia dinh','dong ho','nguoi nha']): labels.append('isFamily')
    if any(c in before for c in ['tien su','truoc day','cu','da tung']) or 'tien su' in _norm(text[:s]): labels.append('isHistorical')
    if 'hien tai' in before.split('.')[-1] or 'theo doi' in before.split('.')[-1]:
        labels=[x for x in labels if x not in {'isHistorical','isNegated'}]
    return {'labels':ordered(labels),'rule_hits':hits}
