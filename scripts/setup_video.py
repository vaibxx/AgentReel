"""Download a local image-to-video model once, outside the repo, and record its licence.

Run on the GPU machine before a Live Run so generation never waits on a download:
    python scripts/setup_video.py --model auto     # picks by this GPU's memory
    python scripts/setup_video.py --model wan      # Wan2.2 TI2V-5B, ~34 GB download, ~24 GB GPU
    python scripts/setup_video.py --model ltx      # LTX-Video 0.9.5, ~25 GB download, ~12-16 GB GPU
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from huggingface_hub import snapshot_download
from agentreel.animate import BACKENDS,choose,vram_gb
from agentreel.illustrate import model_dir

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model',choices=['auto','wan','ltx','both'],default='auto')
    args=p.parse_args()
    names=['wan','ltx'] if args.model=='both' else [choose(args.model)] if args.model=='auto' else [args.model]
    if names==[None]:
        print(f'No usable GPU for image-to-video here ({vram_gb():.1f} GB). Choose --model wan or --model ltx to download anyway.'); return
    target=model_dir(); target.mkdir(parents=True,exist_ok=True); records=[]
    for name in names:
        spec=BACKENDS[name]
        if spec.get('remote'):
            print(f'{name}: runs on a Hugging Face Space; nothing to download. Optional: set HF_TOKEN in .env for a larger free quota.'); continue
        print(f'Fetching {spec["repo"]} into {target} (several GB; resumable if interrupted)',flush=True)
        path=Path(snapshot_download(spec['repo'],cache_dir=str(target),allow_patterns=spec['patterns']))
        size=sum(f.stat().st_size for f in path.rglob('*') if f.is_file())/1e9
        records.append({'backend':name,'repo':spec['repo'],'snapshot':str(path),'gigabytes':round(size,1),'licence':spec['licence'],
                        'licence_files':sorted(str(f.relative_to(path)) for f in path.rglob('*') if 'licen' in f.name.lower())})
    (target/'video_models.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))
    print(f'GPU memory {vram_gb():.1f} GB; VIDEO_MODEL=auto would use: {choose("auto")}')

if __name__=='__main__': main()
