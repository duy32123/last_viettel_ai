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
    enc=tok([e['input_text'] for e in examples], max_length=max_length, truncation=True, padding=padding)
    enc['labels']=[e['labels'] for e in examples]
    return enc
class FloatMultilabelCollator:
    def __init__(self,tok): self.tok=tok
    def __call__(self,features): return features
def assert_float_multilabel_batch(batch):
    if str(batch['labels'].dtype).endswith('int64') or str(batch['labels'].dtype).endswith('long'):
        raise TypeError('labels must be float')
