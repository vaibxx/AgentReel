"""Command-line entry point for reel generation, review and stage reruns."""
import argparse,json,os,sys
from pathlib import Path
from dotenv import load_dotenv
from agentreel.runtime import ROOT

def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8', errors='replace')
    load_dotenv(ROOT/'.env')
    p=argparse.ArgumentParser(description='AgentReel: sentence/YAML to cited, narrated video with free tools.')
    entry=p.add_mutually_exclusive_group(required=True)
    entry.add_argument('--prompt'); entry.add_argument('--brief',type=Path)
    entry.add_argument('--approve',type=Path); entry.add_argument('--rerun-stage',choices=['planner','scriptwriter','storyboarder','critic'])
    entry.add_argument('--doctor',action='store_true')
    entry.add_argument('--record-intervention',type=Path)
    p.add_argument('--stage',default=''); p.add_argument('--change',default=''); p.add_argument('--reason',default='')
    p.add_argument('--run-dir',type=Path); p.add_argument('--reviewer',default='')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--team-id')
    p.add_argument('--duration',type=int,choices=range(30,61)); p.add_argument('--format',choices=['video_9x16','video_16x9'])
    p.add_argument('--brand',choices=['somaiya','none']); p.add_argument('--mode',choices=['assignment','general'])
    p.add_argument('--documents',nargs='+',type=Path,help='Official source PDFs/TXT/MD supplied for this run.')
    p.add_argument('--media-manifest',type=Path); p.add_argument('--no-music',action='store_true'); p.add_argument('--live',action='store_true')
    p.add_argument("--stock-source",choices=["auto","pixabay","none"],default="auto",help="Auto uses Pixabay when configured, for general unbranded videos only.")
    p.add_argument('--video-model',choices=['auto','wan','ltx','hf','none'],default=None,help='Animate illustration stills: wan (local Wan2.2 TI2V-5B, ~24 GB GPU), ltx (local LTX-Video 0.9.5, ~12-16 GB), hf (Wan 2.2 14B on a free Hugging Face Space, a few clips a day), auto picks the best available (default, or VIDEO_MODEL in .env).')
    p.add_argument('--image-backend',choices=['auto','cloudflare','hf','local','none'],default=None,help='Image generator for illustration shots: cloudflare (FLUX.2 klein, free daily quota, needs keys), hf (FLUX.1 schnell Space, free), local (SD-Turbo on a CUDA GPU); auto tries them in that order (default, or IMAGE_BACKEND in .env).')
    p.add_argument('--illustrations',choices=['auto','none'],default='auto',help='Auto paints illustration shots locally with SD-Turbo when a CUDA GPU is available.')
    args=p.parse_args()
    if args.image_backend: os.environ['IMAGE_BACKEND']=args.image_backend
    if args.record_intervention:
        from agentreel.runtime import Run
        Run(args.record_intervention).intervention(args.stage,args.change,args.reason)
        print('Intervention recorded.'); return
    if args.doctor:
        from agentreel.speech import executable
        checks={'pixabay_key_present':bool(os.getenv('PIXABAY_API_KEY')),'python':sys.version.split()[0],'ffmpeg':executable('ffmpeg'),'ffprobe':executable('ffprobe'),'llm_provider':os.getenv('LLM_PROVIDER'),'llm_model':os.getenv('LLM_MODEL'),'key_present':bool(os.getenv('LLM_API_KEY')),'free_access_confirmed':os.getenv('FREE_API_CONFIRMED')=='true','kokoro_model':(ROOT/'.cache/kokoro/model.onnx').is_file(),'kokoro_voices':(ROOT/'.cache/kokoro/voices.bin').is_file(),'fonts':(ROOT/'assets/fonts/NotoSans.ttf').is_file(),'somaiya_corpus':(ROOT/'somaiya_knowledge/rag/chunks.jsonl').is_file(),'illustration_device':__import__('agentreel.illustrate',fromlist=['device']).device(),'illustration_model_dir':str(__import__('agentreel.illustrate',fromlist=['model_dir']).model_dir()),'video_gpu_gb':round(__import__('agentreel.animate',fromlist=['vram_gb']).vram_gb(),1),'video_model':__import__('agentreel.animate',fromlist=['choose']).choose(),'image_backends':__import__('agentreel.imagegen',fromlist=['order']).order(),'cloudflare_keys_present':bool(os.getenv('CLOUDFLARE_ACCOUNT_ID') and os.getenv('CLOUDFLARE_API_TOKEN')),'hf_token_present':bool(os.getenv('HF_TOKEN'))}
        print(json.dumps(checks,indent=2)); return
    from agentreel.pipeline import generate,approve,rerun_stage
    if args.approve: print(approve(args.approve,args.reviewer)); return
    if args.rerun_stage:
        if not args.run_dir: p.error('--rerun-stage requires --run-dir')
        rerun_stage(args.run_dir,args.rerun_stage,args.seed); return
    generate(args)
if __name__=='__main__':
    try: main()
    except (Exception,KeyboardInterrupt) as exc:
        print(f'ERROR: {type(exc).__name__}: {str(exc)[:1800]}',file=sys.stderr)
        raise SystemExit(1)
