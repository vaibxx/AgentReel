import json,os,time
import requests
from .runtime import ROOT,dump,sha

def open_brackets(text,stop):
    """Closers still owed at position stop, ignoring brackets inside strings."""
    stack=[]; quoted=False; escaped=False
    for ch in text[:stop]:
        if quoted:
            if escaped: escaped=False
            elif ch=='\\': escaped=True
            elif ch=='"': quoted=False
        elif ch=='"': quoted=True
        elif ch in '[{': stack.append(']' if ch=='[' else '}')
        elif ch in ']}' and stack: stack.pop()
    return stack

def lenient_json(text):
    """Repair the commonest model slip, an object left unclosed before its next sibling
    or its parent's closing bracket, and unbalanced trailing brackets. Returns normalized
    JSON text; schema validation still decides whether the content is usable."""
    for _ in range(30):
        try: return json.dumps(json.loads(text))
        except json.JSONDecodeError as exc:
            pos=exc.pos; rest=text[pos:].lstrip(); owed=open_brackets(text,pos)
            # A complete document followed by stray closers or commentary: keep the document.
            if exc.msg.startswith('Extra data'): text=text[:pos]; continue
            if exc.msg.startswith('Expecting property name') and rest.startswith('{'):
                comma=text.rfind(',',0,pos)
                if comma<0: raise
                text=text[:comma]+'}'+text[comma:]; continue
            if rest[:1] in (']','}') and owed and owed[-1]!=rest[:1]:
                at=len(text)-len(rest); text=text[:at]+owed[-1]+text[at:]; continue
            if not rest.strip() and owed:
                text=text.rstrip()+''.join(reversed(owed)); continue
            raise
    return text

class Provider:
    def __init__(self,run):
        self.run=run
        self.kind=os.getenv('LLM_PROVIDER','openai_compatible')
        self.model=os.getenv('LLM_MODEL','nvidia/nemotron-3-ultra-550b-a55b')
        self.base=os.getenv('LLM_BASE_URL','https://integrate.api.nvidia.com/v1').rstrip('/')
        self.key=os.getenv('LLM_API_KEY','')
        self.last_request=0.
        if os.getenv('FREE_API_CONFIRMED','false').lower()!='true': raise ValueError('Set FREE_API_CONFIRMED=true only for a free-access account/model.')
        if not self.key and not self.base.startswith(('http://localhost:','http://127.0.0.1:')): raise ValueError('Missing LLM_API_KEY in local .env.')
    def generate(self,agent,schema,data,feedback=''):
        template=(ROOT/'prompts'/f'{agent}.md').read_text(encoding='utf-8')
        system=template+'\nReturn ONLY one JSON object matching this JSON schema:\n'+json.dumps(schema.model_json_schema())
        user=json.dumps(data,ensure_ascii=False)+'\nRevision feedback: '+feedback
        output_limit=16000 if agent=='storyboarder' else 6000
        for attempt in range(3):
            self.run.check()
            if self.kind=='gemini' and self.model=='gemini-3.8-flash':
                time.sleep(max(0,13-(time.monotonic()-self.last_request)))
            if self.run.calls>=int(os.getenv('MAX_LLM_CALLS','24')): raise RuntimeError('Maximum LLM calls reached; stopping instead of spending more requests.')
            self.run.calls+=1; num=self.run.calls
            request_file=dump(self.run.logs/f'calls/{num:03}_input.json',{'system':system,'user':user,'model':self.model,'seed':self.run.seed})
            t=time.monotonic(); response_file=None; usage={}; r=None
            try:
                self.last_request=time.monotonic()
                if self.kind=='gemini':
                    r=requests.post(f'https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent',headers={'x-goog-api-key':self.key},json={'systemInstruction':{'parts':[{'text':system}]},'contents':[{'parts':[{'text':user}]}],'generationConfig':{'temperature':0.35,'maxOutputTokens':output_limit,'responseMimeType':'application/json'}},timeout=(15,240))
                else:
                    r=requests.post(self.base+'/chat/completions',headers={'Authorization':'Bearer '+self.key},json={'model':self.model,'messages':[{'role':'system','content':system},{'role':'user','content':user}],'temperature':0.35,'max_tokens':output_limit,'stream':False,'seed':self.run.seed},timeout=(15,240))
                if not r.ok: raise RuntimeError(f'LLM HTTP {r.status_code}; check provider availability/quota.')
                payload=r.json()
                if self.kind=='gemini':
                    content=''.join(p.get('text','') for p in payload.get('candidates',[{}])[0].get('content',{}).get('parts',[]) if not p.get('thought'))
                    usage=payload.get('usageMetadata',{}); usage={'prompt_tokens':usage.get('promptTokenCount',0),'completion_tokens':usage.get('candidatesTokenCount',0)}
                else:
                    content=payload['choices'][0]['message'].get('content') or ''; usage=payload.get('usage',{})
                response_file=dump(self.run.logs/f'calls/{num:03}_output.json',{'content':content,'usage':usage})
                stripped=content.strip()
                if stripped.startswith('```'): stripped=stripped.split('\n',1)[1].rsplit('```',1)[0].strip()
                try: result=schema.model_validate_json(stripped)
                except Exception:
                    result=schema.model_validate_json(lenient_json(stripped))
                self.run.emit(agent,'llm',provider=self.kind,model=self.model,prompt_id=agent+'_v1',input_tokens=usage.get('prompt_tokens',0),output_tokens=usage.get('completion_tokens',0),latency_s=round(time.monotonic()-t,3),input_path=str(request_file.relative_to(self.run.path)),input_sha256=sha(request_file),output_path=str(response_file.relative_to(self.run.path)),output_sha256=sha(response_file),cost_basis='user_confirmed_free_access; account billing not observable')
                return result
            except Exception as exc:
                self.run.emit(agent,'llm','error',provider=self.kind,model=self.model,prompt_id=agent+'_v1',input_tokens=usage.get('prompt_tokens',0),output_tokens=usage.get('completion_tokens',0),latency_s=round(time.monotonic()-t,3),error_type=type(exc).__name__,http_status=r.status_code if r is not None else None,input_path=str(request_file.relative_to(self.run.path)),output_path=str(response_file.relative_to(self.run.path)) if response_file else None)
                if attempt==2: raise
                if response_file:
                    # A repair needs the failed document, not only an error referring
                    # to array indexes absent from the conversation.
                    user=json.dumps(data,ensure_ascii=False)+'\nRevision feedback: '+feedback+'\nRepair this previous JSON; preserve valid content and correct ALL validation errors:\n'+content[:100000]+'\nValidation errors:\n'+str(exc)[:5000]
                time.sleep(2*(attempt+1))
