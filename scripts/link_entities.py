from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.kb_schema import read_jsonl
from src.linking.retrieval import LexicalIndex
from src.linking.pipeline import load_jsonl, write_jsonl, link_rows

def main():
 p=argparse.ArgumentParser(); p.add_argument('input'); p.add_argument('--output', required=True); p.add_argument('--config', default='configs/linking.yaml'); ns=p.parse_args(); cfg=json.loads(Path(ns.config).read_text(encoding="utf-8"))
 root=Path(cfg.get('index_dir') or cfg['kb_dir']); records=[]
 for pth in root.glob('*.jsonl'): records.extend(read_jsonl(pth))
 linked=link_rows(load_jsonl(ns.input), LexicalIndex(records, bool(cfg.get('include_unverified', False))), int(cfg.get('top_k',5))); write_jsonl(ns.output, linked); print(json.dumps({'records':len(linked),'output':ns.output}, ensure_ascii=False))
if __name__=='__main__': main()
