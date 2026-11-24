import hashlib,ipaddress,json,re,socket
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from .runtime import ROOT,now,dump

def norm(s): return re.sub(r'\s+',' ',s).strip()
def official(url):
    host=urlsplit(url).hostname or ''
    return host=='somaiya.edu' or host.endswith(('.somaiya.edu','.somaiya.edu.in','.gov.in'))
def safe_url(url):
    p=urlsplit(url)
    if p.scheme not in ('http','https') or not p.hostname or p.username or p.password: raise ValueError('Only public HTTP(S) source URLs are accepted.')
    addresses=socket.getaddrinfo(p.hostname,p.port or (443 if p.scheme=='https' else 80),type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses): raise ValueError('Private/local source URLs are not allowed.')
    return url

BLOCKED_PAGE=re.compile(r'client challenge|just a moment|attention required|access denied|verify you are (a )?human|enable javascript|are you a robot|captcha|request blocked',re.I)

def wikipedia(run,query):
    """Encyclopedic fallback via the public MediaWiki API (free, no key)."""
    api='https://en.wikipedia.org/w/api.php'; headers={'User-Agent':'AgentReelAcademic/1.0 (educational video research)'}
    def get():
        found=requests.get(api,params=dict(action='query',list='search',srsearch=query,srlimit=2,format='json'),headers=headers,timeout=20).json()
        titles=[h['title'] for h in found.get('query',{}).get('search',[])]
        pages=[]
        for title in titles:
            data=requests.get(api,params=dict(action='query',prop='extracts',explaintext=1,titles=title,format='json',redirects=1),headers=headers,timeout=20).json()
            for page in data.get('query',{}).get('pages',{}).values():
                if len(page.get('extract',''))>400:
                    pages.append({'url':'https://en.wikipedia.org/wiki/'+page['title'].replace(' ','_'),'title':page['title'],'text':norm(page['extract']),'retrieved_at':now(),'page':None,'approved_source':False})
        return pages
    return run.tool('researcher','wikipedia',get,{'query':query})

def fetch_text(url):
    for _ in range(5):
        safe_url(url)
        r=requests.get(url,timeout=30,allow_redirects=False,stream=True,headers={'User-Agent':'AgentReelAcademic/1.0'})
        if r.is_redirect:
            from urllib.parse import urljoin
            url=urljoin(url,r.headers['Location']); r.close(); continue
        r.raise_for_status(); data=bytearray()
        for part in r.iter_content(65536):
            data.extend(part)
            if len(data)>5_000_000: raise ValueError('Source exceeds 5 MB fetch limit; download and ingest it explicitly.')
        if data.startswith(b'%PDF'):
            import io
            text='\n'.join(p.extract_text() or '' for p in PdfReader(io.BytesIO(data)).pages)
            title=urlsplit(url).path.rsplit('/',1)[-1]
        else:
            soup=BeautifulSoup(bytes(data),'lxml'); title=soup.title.get_text(' ',strip=True) if soup.title else url
            for el in soup.select('script,style,nav,header,footer,form,noscript'): el.decompose()
            text=soup.get_text(' ',strip=True)
        if len(text)<150: raise ValueError('No usable source text; JavaScript, OCR or another source may be needed.')
        # Bot walls return 200 with a stub page; treat them as unreadable, never as evidence.
        if BLOCKED_PAGE.search(title+' '+text[:600]) and len(text.split())<400: raise ValueError('Source returned a bot-check page instead of content.')
        return {'url':url,'title':title,'text':norm(text),'retrieved_at':now(),'page':None,'approved_source':False}
    raise ValueError('Too many redirects')

def uploaded_chunks(paths):
    rows=[]
    for path in paths:
        path=Path(path).resolve(); digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix.lower()=='.pdf': pages=[(i+1,p.extract_text() or '') for i,p in enumerate(PdfReader(path).pages)]
        elif path.suffix.lower() in ('.txt','.md'): pages=[(None,path.read_text(encoding='utf-8'))]
        else: raise ValueError('Reference documents must be PDF, TXT or Markdown.')
        for page,text in pages:
            words=text.split()
            for start in range(0,len(words),300):
                piece=' '.join(words[start:start+380])
                if len(piece)<80: continue
                rows.append({'source_id':'upload_'+digest[:16],'chunk_id':f'upload_{digest[:16]}_{page}_{start}','title':path.name,'text':piece,'url':path.as_uri(),'page':page,'section':f'Provided document page {page}','retrieved_at':now(),'approved_source':True,'content_sha256':digest})
    if paths and not rows: raise ValueError('Uploaded documents have no extractable text. OCR the scans first.')
    return rows

class Knowledge:
    def __init__(self,run,brand,paths=()):
        self.run=run
        self.rows=run.tool('researcher','read_provided_documents',lambda:uploaded_chunks(paths),{'files':[str(p) for p in paths]})
        if brand=='somaiya':
            path=ROOT/'somaiya_knowledge/rag/chunks.jsonl'
            if not path.exists(): raise FileNotFoundError('Somaiya corpus missing: restore somaiya_knowledge/rag/chunks.jsonl.')
            self.rows.extend(run.tool('researcher','load_somaiya_corpus',lambda:[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()],{'path':str(path)}))
        self.vectorizer=None; self.matrix=None; self.semantic=None
        if self.rows:
            def build():
                from scipy.sparse import save_npz
                v=TfidfVectorizer(ngram_range=(1,2),max_features=65000,sublinear_tf=True,strip_accents='unicode')
                texts=[(r.get('title','')+' ')*2+r.get('section','')+' '+r['text'] for r in self.rows]
                matrix=v.fit_transform(texts)
                folder=run.path/'retrieval'; folder.mkdir(exist_ok=True)
                save_npz(folder/'vectors.npz',matrix)
                dump(folder/'vector_metadata.json',{'method':'TF-IDF sparse cosine vector retrieval','dimensions':matrix.shape[1],'documents':matrix.shape[0],'source_ids':[r.get('chunk_id','') for r in self.rows]})
                return v,matrix
            self.vectorizer,self.matrix=run.tool('researcher','build_lexical_index',build,{'documents':len(self.rows)})
            from .embeddings import SemanticStore,MODEL
            self.semantic=run.tool('researcher','load_or_build_faiss',lambda:SemanticStore(self.rows,run.check),{'documents':len(self.rows),'model':MODEL})
            run.artifact('retrieval/semantic_metadata.json',{'backend':'FAISS IndexFlatIP','model':MODEL,'dimensions':self.semantic.index.d,'documents':self.semantic.index.ntotal,'fingerprint':self.semantic.fingerprint,'cache':str(self.semantic.folder)},'researcher')
    def retrieve(self,queries,k=18):
        if self.matrix is None: return []
        queries=list(queries)
        if any(re.search(r'\b(startups?|entrepreneur\w*|incubat\w*)\b',q,re.I) for q in queries):
            queries.append('startup startups entrepreneurship incubation incubator')
        def go():
            scores=np.asarray((self.matrix@self.vectorizer.transform(queries).T).toarray()).max(axis=1)
            semantic=self.semantic.scores(queries)
            # Hybrid retrieval retains exact programme names alongside semantic matches.
            scores=.65*np.maximum(semantic,0)+.35*scores
            scores+=np.array([.1 if r.get('approved_source') and scores[i]>=.035 else 0 for i,r in enumerate(self.rows)])
            selected=[]; per_source={}
            for i in np.argsort(-scores):
                if scores[i]<0.035: break
                r=self.rows[i]; key=r.get('source_id',r['url'])
                if per_source.get(key,0)>=2: continue
                score=float(scores[i])
                selected.append({**r,'retrieval_score':score}); per_source[key]=per_source.get(key,0)+1
                if len(selected)>=k: break
            return selected
        return self.run.tool('researcher','vector_search',go,{'queries':queries,'top_k':k})

def web_research(run,queries,assignment=False):
    from ddgs import DDGS
    candidates=[]
    for query in queries[:4]:
        q=query+(' site:somaiya.edu' if assignment else '')
        try:
            hits=run.tool('researcher','web_search',lambda:list(DDGS(timeout=20).text(q,max_results=5)),{'query':q})
            candidates.extend(hit.get('href','') for hit in hits)
        except Exception: continue
    records=[]; pages=[]
    for url in dict.fromkeys(candidates):
        if not url or (assignment and not official(url)): continue
        try:
            pages.append(run.tool('researcher','fetch_source',lambda:fetch_text(url),{'url':url}))
        except Exception: continue
        if len(pages)>=6: break
    # Search and fetches fail often (bot walls, timeouts); general videos top up from Wikipedia.
    if not assignment and len(pages)<3:
        for query in queries[:2]:
            try: pages.extend(wikipedia(run,query))
            except Exception: continue
    seen=set()
    for r in pages:
        if r['url'] in seen: continue
        seen.add(r['url'])
        try:
            if assignment and not official(r['url']): continue
            r['source_id']='web_'+hashlib.sha256(r['url'].encode()).hexdigest()[:16]
            # Select passages by relevance instead of sending the whole site to an LLM.
            words=r.pop('text').split(); pieces=[' '.join(words[i:i+350]) for i in range(0,len(words),300)]
            v=TfidfVectorizer().fit(pieces+queries)
            scores=(v.transform(pieces)@v.transform(queries).T).toarray().max(axis=1)
            for i in np.argsort(-scores)[:2]: records.append({**r,'text':pieces[i],'chunk_id':r['source_id']+'_'+str(i),'section':'Fetched web passage'})
            if len(records)>=12: break
        except Exception: continue
    if not records: raise RuntimeError('Web research returned no readable sources. Supply official PDFs/TXT with --documents, or retry when search is available.')
    return records
