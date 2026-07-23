def _prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0; f=2*p*r/(p+r) if p+r else 0.0
    return {'tp':tp,'fp':fp,'fn':fn,'precision':p,'recall':r,'f1':f}
def evaluate_spans(gold, pred):
    g=set(); p=set(); gb=set(); pb=set(); types=set(); boundary_correct=type_ok=0
    for r in gold:
        for e in r.get('entities',[]):
            if e.get('type') in {'IGNORE','UNMAPPED'}: continue
            s=e.get('start', e.get('position',[None,None])[0]); en=e.get('end', e.get('position',[None,None])[1]); typ=e.get('type'); g.add((r.get('id'),s,en,typ)); gb.add((r.get('id'),s,en)); types.add(typ)
    for r in pred:
        for e in r.get('entities',[]):
            s=e.get('start', e.get('position',[None,None])[0]); en=e.get('end', e.get('position',[None,None])[1]); typ=e.get('type'); p.add((r.get('id'),s,en,typ)); pb.add((r.get('id'),s,en)); types.add(typ)
            if (r.get('id'),s,en) in gb:
                boundary_correct += 1
                if (r.get('id'),s,en,typ) in g: type_ok += 1
    tp=len(g&p); fp=len(p-g); fn=len(g-p)
    per_type={}
    for t in types:
        gt={x for x in g if x[3]==t}; pt={x for x in p if x[3]==t}; per_type[t]=_prf(len(gt&pt),len(pt-gt),len(gt-pt))
    return {'strict_micro':_prf(tp,fp,fn),'false_positives':fp,'false_negatives':fn,'boundary_only':{'tp':len(gb&pb)},'type_accuracy_on_correct_boundary':(type_ok/boundary_correct if boundary_correct else 0.0),'per_type':per_type}
