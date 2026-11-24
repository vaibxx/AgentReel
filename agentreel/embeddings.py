"""Local sentence-transformers all-MiniLM-L6-v2 ONNX encoder + persistent FAISS.

The ONNX export uses masked mean pooling and unit normalization, as in the
original SentenceTransformer model. No embedding API or GPU is required.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from .runtime import ROOT,dump,sha

MODEL='sentence-transformers/all-MiniLM-L6-v2 (Xenova quantized ONNX export)'
class Encoder:
    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        folder=ROOT/'.cache/embeddings'
        if not (folder/'model.onnx').exists(): raise FileNotFoundError('Run python scripts/setup_models.py to install the free embedding model.')
        self.tokenizer=Tokenizer.from_file(str(folder/'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding()
        options=ort.SessionOptions(); options.intra_op_num_threads=4
        self.session=ort.InferenceSession(str(folder/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.names={item.name for item in self.session.get_inputs()}
        self.fingerprint=sha(folder/'model.onnx')+sha(folder/'tokenizer.json')
    def encode(self,texts,check=lambda:None):
        batches=[]
        for start in range(0,len(texts),24):
            check(); encoded=self.tokenizer.encode_batch(texts[start:start+24])
            ids=np.asarray([e.ids for e in encoded],dtype=np.int64)
            mask=np.asarray([e.attention_mask for e in encoded],dtype=np.int64)
            feed={'input_ids':ids,'attention_mask':mask,'token_type_ids':np.zeros_like(ids)}
            hidden=self.session.run(None,{k:v for k,v in feed.items() if k in self.names})[0]
            vectors=(hidden*mask[:,:,None]).sum(axis=1)/np.maximum(mask.sum(axis=1,keepdims=True),1)
            vectors=vectors/np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
            batches.append(vectors.astype(np.float32))
        return np.concatenate(batches)

class SemanticStore:
    def __init__(self,rows,check=lambda:None):
        import faiss
        self.rows=rows; self.encoder=Encoder(); faiss.omp_set_num_threads(4)
        texts=[r.get('title','')+' '+r.get('section','')+' '+r['text'] for r in rows]
        fingerprint=hashlib.sha256((self.encoder.fingerprint+'\n'.join(texts)).encode()).hexdigest()
        self.folder=ROOT/'.cache/faiss'/fingerprint; self.folder.mkdir(parents=True,exist_ok=True)
        path=self.folder/'index.faiss'
        if path.exists(): self.index=faiss.read_index(str(path))
        else:
            vectors=self.encoder.encode(texts,check)
            self.index=faiss.IndexFlatIP(vectors.shape[1]); self.index.add(vectors)
            temporary=self.folder/'index.tmp'; faiss.write_index(self.index,str(temporary)); temporary.replace(path)
            dump(self.folder/'metadata.json',{'model':MODEL,'dimensions':vectors.shape[1],'documents':len(rows),'fingerprint':fingerprint,'source_ids':[r.get('chunk_id','') for r in rows]})
        self.fingerprint=fingerprint
    def scores(self,queries):
        vectors=self.encoder.encode(queries)
        distances,indices=self.index.search(vectors,len(self.rows))
        result=np.full(len(self.rows),-1.,dtype=np.float32)
        for d,ix in zip(distances,indices): np.maximum.at(result,ix,d)
        return result
