from __future__ import annotations

import csv
import hashlib
import json
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .kb_schema import KBRecord

RX_TTYS={"IN","PIN","MIN","BN","SCD","SBD","SCDC","SBDC","SCDF","SBDF","GPCK","BPCK","DF"}
INGREDIENT_TTYS={"IN","PIN","MIN"}
BRAND_TTYS={"BN"}
PRODUCT_TTYS={"SCD","SBD","SCDC","SBDC","SCDF","SBDF","GPCK","BPCK"}
TTY_PRECEDENCE=("IN","PIN","MIN","BN","SCD","SBD","SCDC","SBDC","SCDF","SBDF","GPCK","BPCK","DF")
NLM_ATTRIBUTION="RxNorm Current Prescribable Content, U.S. National Library of Medicine"
RXNCONSO_FIELDS=("RXCUI","LAT","TS","LUI","STT","SUI","ISPREF","RXAUI","SAUI","SCUI","SDUI","SAB","TTY","CODE","STR","SRL","SUPPRESS","CVF")
RXNREL_FIELDS=("RXCUI1","RXAUI1","STYPE1","REL","RXCUI2","RXAUI2","STYPE2","RELA","RUI","SRUI","SAB","SL","RG","DIR","SUPPRESS","CVF")

def file_sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def rrf_member_sha256(path: Path, filename: str='RXNCONSO.RRF') -> str|None:
    h=hashlib.sha256()
    if path.is_dir():
        p=path/filename
        if not p.exists(): return None
        with p.open('rb') as fh:
            for chunk in iter(lambda: fh.read(1024*1024), b''): h.update(chunk)
        return h.hexdigest()
    if path.suffix.lower()=='.zip':
        name=_zip_member(path, filename)
        if not name: return None
        with zipfile.ZipFile(path) as z, z.open(name) as fh:
            for chunk in iter(lambda: fh.read(1024*1024), b''): h.update(chunk)
        return h.hexdigest()
    return file_sha256(path)

def _zip_member(zip_path: Path, suffix: str) -> str|None:
    with zipfile.ZipFile(zip_path) as z:
        matches=[n for n in z.namelist() if n.endswith(suffix)]
        return sorted(matches)[0] if matches else None

def _iter_rrf_lines(path: Path, filename: str="RXNCONSO.RRF"):
    if path.is_dir():
        p=path/filename
        with p.open(encoding='utf-8', errors='strict') as fh:
            for line in fh: yield line.rstrip('\n')
        return
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as z:
            name=_zip_member(path, filename)
            if not name: return
            with z.open(name) as fh:
                for b in fh: yield b.decode('utf-8', errors='strict').rstrip('\n')
        return
    with path.open(encoding='utf-8', errors='strict') as fh:
        for line in fh: yield line.rstrip('\n')

def parse_rxnconso_line(line: str, *, line_number:int|None=None) -> dict[str,str]:
    cols=line.split('|')
    if cols and cols[-1]=='': cols=cols[:-1]
    if len(cols) < len(RXNCONSO_FIELDS):
        where=f" at line {line_number}" if line_number is not None else ""
        raise ValueError(f"malformed RXNCONSO.RRF row{where}: expected {len(RXNCONSO_FIELDS)} fields, got {len(cols)}")
    return dict(zip(RXNCONSO_FIELDS, cols[:len(RXNCONSO_FIELDS)]))

def parse_rxnrel_line(line: str) -> dict[str,str]:
    cols=line.split('|')
    if cols and cols[-1]=='': cols=cols[:-1]
    if len(cols) < len(RXNREL_FIELDS): raise ValueError('malformed RXNREL.RRF row')
    return dict(zip(RXNREL_FIELDS, cols[:len(RXNREL_FIELDS)]))

def _specificity(tty: str) -> str:
    if tty in INGREDIENT_TTYS: return 'ingredient'
    if tty in BRAND_TTYS: return 'brand'
    if tty in PRODUCT_TTYS: return 'product'
    return 'dose_form'

def _canonical(rows: list[dict[str,str]]) -> dict[str,str]:
    def key(row):
        tty_rank=TTY_PRECEDENCE.index(row['TTY']) if row['TTY'] in TTY_PRECEDENCE else 999
        pref=0 if row.get('ISPREF')=='Y' else 1
        return (tty_rank, pref, row['STR'].casefold(), row['RXCUI'])
    return sorted(rows, key=key)[0]

def _read_relationships(path: Path) -> dict[str,list[dict[str,str]]]:
    rels=defaultdict(list)
    try:
        iterator=_iter_rrf_lines(path, 'RXNREL.RRF')
        for line in iterator:
            if not line: continue
            row=parse_rxnrel_line(line)
            if row.get('SAB')!='RXNORM' or row.get('SUPPRESS') not in ('','N'): continue
            if row.get('RXCUI1') and row.get('RXCUI2'):
                rels[row['RXCUI1']].append({'rela':row.get('RELA') or row.get('REL'),'target_rxcui':row['RXCUI2'],'source':'RXNREL.RRF'})
                rels[row['RXCUI2']].append({'rela':row.get('RELA') or row.get('REL'),'target_rxcui':row['RXCUI1'],'source':'RXNREL.RRF'})
    except FileNotFoundError:
        pass
    return dict(rels)

def import_rxnorm_prescribable(path: Path, *, version: str, source_url: str='', source: str='RxNorm Current Prescribable Content', release_date: str|None=None, term_types: set[str]|None=None) -> tuple[list[KBRecord], dict[str,Any]]:
    term_types=term_types or RX_TTYS; path=Path(path); rows_by_code=defaultdict(list); counts=Counter(); malformed=0
    rxnconso_member=_zip_member(path,'RXNCONSO.RRF') if path.suffix.lower()=='.zip' else str(path/'RXNCONSO.RRF' if path.is_dir() else path)
    for i,line in enumerate(_iter_rrf_lines(path,'RXNCONSO.RRF'),1):
        if not line: continue
        try: row=parse_rxnconso_line(line, line_number=i)
        except ValueError:
            malformed += 1; continue
        counts['rows_total'] += 1; counts[f"LAT:{row['LAT']}"] += 1; counts[f"SAB:{row['SAB']}"] += 1; counts[f"SUPPRESS:{row['SUPPRESS']}"] += 1; counts[f"TTY:{row['TTY']}"] += 1
        if row['LAT']!='ENG' or row['SAB']!='RXNORM' or row['SUPPRESS']!='N' or row['TTY'] not in term_types: continue
        rows_by_code[row['RXCUI']].append(row); counts['accepted_rows'] += 1
    rels=_read_relationships(path)
    records=[]; duplicate_aliases=0
    for rxcui,rows in sorted(rows_by_code.items(), key=lambda kv:int(kv[0]) if kv[0].isdigit() else kv[0]):
        canon=_canonical(rows); seen=set(); aliases=[]; prov=[]; tty_values=sorted({r['TTY'] for r in rows}, key=lambda t: TTY_PRECEDENCE.index(t) if t in TTY_PRECEDENCE else 999)
        for row in sorted(rows, key=lambda r:(r['STR'].casefold(), r['TTY'], r['RXAUI'])):
            norm=' '.join(row['STR'].casefold().split())
            if norm in seen: duplicate_aliases += 1; continue
            seen.add(norm)
            if row['STR'] != canon['STR']: aliases.append(row['STR'])
            prov.append({'raw_alias':row['STR'],'normalized_alias':norm,'TTY':row['TTY'],'RXAUI':row['RXAUI'],'SAB':row['SAB'],'LAT':row['LAT'],'source_field':'STR'})
        meta={'TTY':canon['TTY'],'TTY_values':tty_values,'specificity':_specificity(canon['TTY']),'alias_provenance':prov,'relationships':rels.get(rxcui,[]),'source_url':source_url,'release_date':release_date or version,'zip_checksum':file_sha256(path) if path.is_file() else None,'rxnconso_member':rxnconso_member,'rxnconso_checksum':rrf_member_sha256(path,'RXNCONSO.RRF'),'official_kb':True,'verified_source':True,'license_required':False,'nlm_attribution':NLM_ATTRIBUTION,'import_timestamp':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        records.append(KBRecord(rxcui, canon['STR'], aliases, 'RxNorm', version, source, True, meta, 'drug', 'en'))
    report={'source':source,'source_url':source_url,'version':version,'release_date':release_date or version,'official_kb':True,'verified':True,'rows_total':counts['rows_total'],'accepted_rows':counts['accepted_rows'],'concept_count':len(records),'malformed_rows':malformed,'counts_by_tty':{k.split(':',1)[1]:v for k,v in counts.items() if k.startswith('TTY:')},'accepted_tty':dict(Counter(r.metadata['TTY'] for r in records)),'duplicate_aliases':duplicate_aliases,'zip_checksum':file_sha256(path) if path.is_file() else None,'rxnconso_member':rxnconso_member,'rxnconso_checksum':rrf_member_sha256(path,'RXNCONSO.RRF'),'nlm_attribution':NLM_ATTRIBUTION,'license_required':False}
    return records, report

def import_rxnorm_rrf(path: Path, version: str, source: str="RxNorm", verified: bool=True, term_types: set[str]|None=None) -> list[KBRecord]:
    records,_=import_rxnorm_prescribable(path, version=version, source=source, term_types=term_types or RX_TTYS)
    if not verified:
        for r in records: r.verified=False; r.metadata['official_kb']=False
    return records

def import_rxnorm_csv(path: Path, version: str, source: str, verified: bool = True) -> list[KBRecord]:
    records=[]
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            code=(row.get("code") or row.get("rxcui") or row.get("RXCUI") or "").strip()
            aliases=[s.strip() for s in (row.get("aliases") or row.get("synonyms") or "").split("|") if s.strip()]
            name=(row.get("canonical_name") or row.get("preferred_name") or row.get("STR") or "").strip()
            records.append(KBRecord(code,name,aliases,"RxNorm",version,source,verified,{"input_file":str(path),"TTY":row.get("TTY"),"official_kb":bool(verified)},"drug",row.get("language") or row.get("LAT") or "en"))
    return records
