import csv,hashlib,json,os,subprocess,time,uuid
from datetime import datetime,timezone
from pathlib import Path
from contextlib import contextmanager

ROOT=Path(__file__).resolve().parents[1]
def dump(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    return path
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()

class Run:
    def __init__(self,run_dir=None,seed=42):
        self.path=Path(run_dir).resolve() if run_dir else ROOT/'outputs'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        self.path.mkdir(parents=True,exist_ok=True)
        self.logs=self.path/'logs'; self.logs.mkdir(exist_ok=True)
        self.seed=seed; self.start=time.monotonic(); self.limit=int(os.getenv('MAX_RUN_SECONDS','10800'))
        self.logfile=self.logs/'run_log.jsonl'; self.costfile=self.logs/'cost_report.csv'
        self.seq=0; self.previous='0'*64; self.calls=0
        if self.logfile.exists():
            for line in self.logfile.read_text(encoding='utf-8').splitlines():
                row=json.loads(line); self.seq+=1; self.previous=row['record_sha256']
                self.calls+=row.get('tool')=='llm'
        if not self.costfile.exists():
            self.costfile.write_text('stage,provider,model,input_tokens,output_tokens,cost_inr,latency_s,status\n',encoding='utf-8')
        p=self.logs/'interventions.md'
        if not p.exists(): p.write_text('# Human interventions\n\nNo interventions recorded. Mandatory safety approval is recorded separately.\n',encoding='utf-8')
    def check(self):
        if time.monotonic()-self.start>self.limit: raise TimeoutError('Three-hour run limit reached; no export allowed.')
    def intervention(self,stage,change,reason):
        if not all(x.strip() for x in (stage,change,reason)):
            raise ValueError('Stage, change and reason are required.')
        count=sum(bool(json.loads(line).get('human_intervention')) for line in self.logfile.read_text(encoding='utf-8').splitlines()) if self.logfile.exists() else 0
        if count>=5: raise ValueError('Maximum five human interventions reached.')
        self.emit(stage,'human_change',human_intervention=True,change=change,reason=reason,intervention_number=count+1)
        with (self.logs/'interventions.md').open('a',encoding='utf-8') as f:
            f.write(f'\n## Intervention {count+1} ({now()})\nStage: {stage}\n\nChange: {change}\n\nReason: {reason}\n')
    def emit(self,agent,tool,status='ok',**fields):
        self.seq+=1
        row={'ts':now(),'run_id':self.path.name,'sequence':self.seq,'agent':agent,'tool':tool,'model':'local','input_tokens':0,'output_tokens':0,'cost_inr':0,'latency_s':0,'status':status,'human_intervention':False,'previous_sha256':self.previous,**fields}
        if 'inputs' in fields: row['input_sha256']=hashlib.sha256(json.dumps(fields['inputs'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        encoded=json.dumps(row,sort_keys=True,ensure_ascii=False).encode()
        row['record_sha256']=hashlib.sha256(encoded).hexdigest(); self.previous=row['record_sha256']
        with self.logfile.open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        with self.costfile.open('a',encoding='utf-8',newline='') as f:
            csv.writer(f).writerow([agent,fields.get('provider','local'),row['model'],row['input_tokens'],row['output_tokens'],row['cost_inr'],row['latency_s'],status])
    def artifact(self,name,value,agent):
        path=dump(self.path/name,value)
        snapshot=dump(self.logs/f'artifacts/{self.seq+1:04}_{Path(name).name}',value)
        self.emit(agent,'write_artifact',output_path=str(snapshot.relative_to(self.path)),current_path=str(path.relative_to(self.path)),output_sha256=sha(snapshot))
        return path
    def tool(self,agent,name,fn,inputs=None):
        self.check(); t=time.monotonic()
        try:
            result=fn()
            self.emit(agent,name,latency_s=round(time.monotonic()-t,3),inputs=inputs)
            return result
        except Exception as exc:
            # Never include HTTP request objects/headers or environment variables.
            self.emit(agent,name,'error',latency_s=round(time.monotonic()-t,3),error_type=type(exc).__name__,inputs=inputs)
            raise
    def command(self,agent,args,cwd=None):
        remaining=max(1,int(self.limit-(time.monotonic()-self.start)))
        def go():
            r=subprocess.run([str(a) for a in args],cwd=cwd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=remaining)
            if r.returncode: raise RuntimeError(f'{Path(str(args[0])).name} failed: '+r.stderr[-2500:])
            return r
        return self.tool(agent,'subprocess',go,{'argv':[str(a) for a in args]})

def verify_log(path):
    previous='0'*64; count=0
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        row=json.loads(line); recorded=row.pop('record_sha256')
        if row['previous_sha256']!=previous or hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()!=recorded:
            raise ValueError('Log integrity failure')
        previous=recorded; count+=1
    return count
