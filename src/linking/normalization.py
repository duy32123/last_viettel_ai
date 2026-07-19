from __future__ import annotations
import re, unicodedata
ABBR={"đtđ":"đái tháo đường","dtd":"đái tháo đường","tha":"tăng huyết áp"}
DRUG_UNITS=r"(?:mg|mcg|g|ml|iu|units?|meq)"

def strip_diacritics(text:str)->str:
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c)!='Mn')

def normalize_text(text:str, no_diacritic:bool=False)->str:
    t=unicodedata.normalize('NFC', text).casefold(); t=t.replace('‐','-').replace('‑','-').replace('–','-').replace('—','-').replace('/',' / ')
    t=re.sub(r"[,:;()\[\]{}]", " ", t); t=re.sub(r"\s+", " ", t).strip()
    t=ABBR.get(strip_diacritics(t), t)
    return strip_diacritics(t) if no_diacritic else t

def normalize_mention(text:str, entity_type:str|None=None)->dict:
    original=unicodedata.normalize('NFC', text); norm=normalize_text(original); base=norm; ctx={}
    if entity_type == 'THUỐC':
        m=re.search(rf"\b(\d+(?:[,.]\d+)?)\s*({DRUG_UNITS})\b", norm)
        if m: ctx['strength']=m.group(0).replace(',', '.')
        base=re.sub(rf"\b\d+(?:[,.]\d+)?\s*{DRUG_UNITS}\b", " ", norm)
        base=re.sub(r"\b(?:uống|tiêm|truyền|po|iv|im|sc|bid|tid|qd|daily|sáng|tối)\b", " ", base)
        base=re.sub(r"\s+", " ", base).strip()
    return {'original':original,'normalized':norm,'no_diacritic':normalize_text(original, True),'base':base,'base_no_diacritic':normalize_text(base, True),'features':ctx}
