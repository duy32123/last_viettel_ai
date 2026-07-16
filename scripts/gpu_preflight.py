from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--out', default=None)
    ns=p.parse_args()
    report={'python': sys.version.split()[0]}
    try:
        import torch
        report['torch_version']=torch.__version__
        report['cuda_available']=bool(torch.cuda.is_available())
        report['cuda_version']=getattr(torch.version, 'cuda', None)
        report['bf16_supported']=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        report['fp16_supported']=bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            idx=torch.cuda.current_device(); props=torch.cuda.get_device_properties(idx)
            free,total=torch.cuda.mem_get_info(idx)
            report.update({'gpu_name':props.name,'gpu_total_vram_bytes':int(total),'gpu_free_vram_bytes':int(free)})
        else:
            report.update({'gpu_name':None,'gpu_total_vram_bytes':0,'gpu_free_vram_bytes':0})
    except Exception as e:
        report.update({'torch_version':None,'cuda_available':False,'torch_error':repr(e),'bf16_supported':False,'fp16_supported':False,'gpu_name':None,'gpu_total_vram_bytes':0,'gpu_free_vram_bytes':0})
    try:
        import transformers; report['transformers_version']=transformers.__version__
    except Exception as e: report['transformers_version']=None; report['transformers_error']=repr(e)
    try:
        import accelerate; report['accelerate_version']=accelerate.__version__
    except Exception as e: report['accelerate_version']=None; report['accelerate_error']=repr(e)
    usage=shutil.disk_usage(Path.cwd())
    report['disk_free_bytes']=int(usage.free); report['disk_total_bytes']=int(usage.total)
    text=json.dumps(report, ensure_ascii=False, indent=2)
    if ns.out:
        Path(ns.out).parent.mkdir(parents=True, exist_ok=True); Path(ns.out).write_text(text, encoding='utf-8')
    print(text)

if __name__ == '__main__': main()
