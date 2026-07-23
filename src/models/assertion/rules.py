from __future__ import annotations
import re, unicodedata
from .labels import ordered

def _norm(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s.lower()) if unicodedata.category(c) != 'Mn').replace('đ','d')

def _clause_before(text: str, start: int) -> tuple[str,int]:
    left=max(text.rfind('.',0,start), text.rfind('\n',0,start), text.rfind(';',0,start)) + 1
    return text[left:start], left

def rule_assertions(text, entity):
    s,e=entity['position']; mention=text[s:e]
    clause,base=_clause_before(text,s); before=_norm(clause); full_before=_norm(text[:s])
    labels=[]; hits=[]
    # Negation: only same clause, and stop at contrastive cues before the entity.
    contrast=max(before.rfind(' nhung '), before.rfind(' nhưng '), before.rfind(' ma con '), before.rfind(' mà còn '))
    neg_region=before[contrast+1:] if contrast >= 0 else before
    if 'khong nhung' not in before and any(c in neg_region for c in ['khong ghi nhan','khong dung','khong co','khong ', 'chua thay','chua co','phu nhan','loai tru']):
        labels.append('isNegated')
        m=re.search(r'kh[oô]ng ghi nh[aậ]n|Kh[oô]ng ghi nh[aậ]n', clause)
        if m: hits.append({'rule_id':'neg_scope','cue_span':[base+m.start(), base+m.end()]})
        else: hits.append({'rule_id':'neg_scope','cue_span':[base, min(base+14, s)]})
    # Family cues: family terms before mention, including standalone "gia đình ghi nhận".
    if any(c in before for c in ['me ', 'bo ', 'anh ruot', 'chi ruot', 'em ruot', 'gia dinh', 'dong ho', 'nguoi nha']):
        labels.append('isFamily')
    # Historical cues: patient history or family history before mention; section applies until HIỆN TẠI.
    if any(c in before for c in ['tien su', 'truoc day', 'da tung', 'ho so cu', 'cu ghi']) or ('tien su' in full_before and 'hien tai' not in _norm(text[:s]).split('tien su')[-1]):
        labels.append('isHistorical')
    # Current/follow-up contexts are not assertions by themselves.
    if any(c in before for c in ['hien tai', 'theo doi']) and not any(c in before for c in ['tien su gia dinh','gia dinh','me ','bo ']):
        labels=[x for x in labels if x not in {'isHistorical','isNegated'}]
    return {'labels':ordered(labels),'rule_hits':hits}
