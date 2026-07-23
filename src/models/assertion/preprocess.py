from __future__ import annotations
from .labels import to_vector
SPECIAL_TOKENS=['<ENT_START>','<ENT_END>']+[f'<TYPE_{t}>' for t in ['TRIỆU_CHỨNG','TÊN_XÉT_NGHIỆM','KẾT_QUẢ_XÉT_NGHIỆM','CHẨN_ĐOÁN','THUỐC']]

def make_examples(records, max_chars=256):
    out=[]
    for r in records:
        for e in r.get('entities',[]):
            s,en=e['position']; start=max(0,s-max_chars//2); end=min(len(r['text']),en+max_chars//2)
            ctx=r['text'][start:s] + '<ENT_START> '+e['text']+' <ENT_END>' + r['text'][en:end]
            out.append({'input_text':f'<TYPE_{e["type"]}> '+ctx,'labels':to_vector(e.get('assertions',[]))})
    return out

def register_special_tokens(tok, model=None):
    added=tok.add_special_tokens({'additional_special_tokens':SPECIAL_TOKENS})
    if model is not None: model.resize_token_embeddings(len(tok))
    return added

def tokenize_examples(examples, tok, max_length=256, padding=True):
    texts=[e['input_text'] for e in examples]
    enc=tok(texts, max_length=max_length, truncation=True, padding=padding)
    enc['labels']=[e['labels'] for e in examples]
    return enc

class FloatMultilabelCollator:
    def __init__(self,tok): self.tok=tok
    def __call__(self,features):
        import torch
        labels=[f.pop('labels') for f in features]
        batch=self.tok.pad(features, padding=True, return_tensors='pt') if hasattr(self.tok,'pad') else _pad_features(features)
        batch['labels']=torch.tensor(labels, dtype=torch.float32)
        assert_float_multilabel_batch(batch)
        return batch

def _pad_features(features):
    import torch
    max_len=max(len(f['input_ids']) for f in features) if features else 0
    input_ids=[]; attention=[]
    for f in features:
        pad=max_len-len(f['input_ids'])
        input_ids.append(list(f['input_ids'])+[0]*pad)
        attention.append(list(f.get('attention_mask',[1]*len(f['input_ids'])))+[0]*pad)
    return {'input_ids':torch.tensor(input_ids,dtype=torch.long),'attention_mask':torch.tensor(attention,dtype=torch.long)}

def assert_float_multilabel_batch(batch):
    if str(batch['labels'].dtype).endswith('int64') or str(batch['labels'].dtype).endswith('long'):
        raise TypeError('labels must be float')
