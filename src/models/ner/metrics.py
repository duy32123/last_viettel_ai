def evaluate_spans(gold, pred):
    g=set(); p=set(); gb=set(); pb=set(); boundary_correct=type_ok=0
    for r in gold:
        for e in r.get('entities',[]):
            if e.get('type') in {'IGNORE','UNMAPPED'}: continue
            s=e.get('start', e.get('position',[None,None])[0]); en=e.get('end', e.get('position',[None,None])[1]); g.add((r.get('id'),s,en,e.get('type'))); gb.add((r.get('id'),s,en))
    for r in pred:
        for e in r.get('entities',[]):
            s=e.get('start', e.get('position',[None,None])[0]); en=e.get('end', e.get('position',[None,None])[1]); p.add((r.get('id'),s,en,e.get('type'))); pb.add((r.get('id'),s,en))
            if (r.get('id'),s,en) in gb:
                boundary_correct += 1
                if (r.get('id'),s,en,e.get('type')) in g: type_ok += 1
    tp=len(g&p); fp=len(p-g); fn=len(g-p); btp=len(gb&pb)
    return {'strict_micro':{'tp':tp,'fp':fp,'fn':fn},'false_positives':fp,'false_negatives':fn,'boundary_only':{'tp':btp},'type_accuracy_on_correct_boundary':(type_ok/boundary_correct if boundary_correct else 0.0)}
