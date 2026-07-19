from __future__ import annotations
import csv, zipfile
from pathlib import Path
from .kb_schema import KBRecord

RX_TTYS={"IN","PIN","MIN","BN","SCD","SBD"}

def _rows_from_rrf(path: Path):
    if path.is_dir():
        p=path/"RXNCONSO.RRF"; yield from p.read_text(encoding="utf-8", errors="ignore").splitlines(); return
    if path.suffix.lower()==".zip":
        with zipfile.ZipFile(path) as z:
            name=next(n for n in z.namelist() if n.endswith("RXNCONSO.RRF"))
            for b in z.open(name): yield b.decode("utf-8", errors="ignore").rstrip("\n")
        return
    yield from path.read_text(encoding="utf-8", errors="ignore").splitlines()

def import_rxnorm_rrf(path: Path, version: str, source: str="RxNorm", verified: bool=True, term_types: set[str]|None=None) -> list[KBRecord]:
    term_types=term_types or RX_TTYS; by_code={}
    for line in _rows_from_rrf(path):
        cols=line.split("|")
        if len(cols)<17: continue
        rxcui,lat,sab,tty,code,str_,suppress = cols[0], cols[1], cols[11], cols[12], cols[13], cols[14], cols[16]
        cvf=cols[17] if len(cols)>17 else ""
        if sab!="RXNORM" or suppress!="N" or tty not in term_types: continue
        rec=by_code.get(rxcui)
        meta={"TTY":tty,"LAT":lat,"SAB":sab,"CODE":code,"CVF":cvf,"input_file":str(path)}
        if not rec:
            by_code[rxcui]=KBRecord(rxcui,str_,[],"RxNorm",version,source,verified,meta,"drug",lat.lower())
        else:
            rec.aliases.append(str_); rec.metadata.setdefault("TTY", tty)
    return list(by_code.values())

def import_rxnorm_csv(path: Path, version: str, source: str, verified: bool = True) -> list[KBRecord]:
    records=[]
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            code=(row.get("code") or row.get("rxcui") or row.get("RXCUI") or "").strip()
            aliases=[s.strip() for s in (row.get("aliases") or row.get("synonyms") or "").split("|") if s.strip()]
            name=(row.get("canonical_name") or row.get("preferred_name") or row.get("STR") or "").strip()
            records.append(KBRecord(code,name,aliases,"RxNorm",version,source,verified,{"input_file":str(path),"TTY":row.get("TTY")},"drug",row.get("language") or row.get("LAT") or "en"))
    return records
