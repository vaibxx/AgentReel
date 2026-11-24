"""Download local illustration models once, outside the repo, and record their licences.

Run before a Live Run so generation never waits on a download.
"""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from huggingface_hub import snapshot_download
from agentreel.illustrate import SD_REPO,DEPTH_REPO,model_dir,device

PATTERNS={SD_REPO:['*.json','*.txt','*.md','*fp16.safetensors','tokenizer/*','scheduler/*'],
          DEPTH_REPO:['*.json','*.md','*.safetensors']}

def main():
    target=model_dir(); target.mkdir(parents=True,exist_ok=True); records=[]
    for repo,patterns in PATTERNS.items():
        print('Fetching',repo,'into',target,flush=True)
        path=Path(snapshot_download(repo,cache_dir=str(target),allow_patterns=patterns))
        licences=sorted(str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and 'licen' in p.name.lower())
        card=(path/'README.md').read_text(encoding='utf-8',errors='replace') if (path/'README.md').is_file() else ''
        declared=next((line.split(':',1)[1].strip() for line in card.splitlines()[:30] if line.lower().startswith('license')),'see model card')
        records.append({'repo':repo,'snapshot':str(path),'licence_declared':declared,'licence_files':licences})
    (target/'illustration_models.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2)); print('Device available for generation:',device())

if __name__=='__main__': main()
