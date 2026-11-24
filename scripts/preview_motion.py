"""Re-render a completed nonsensitive run with the new engine, without API calls.

Creates a NEW run so the previous deliverable and audit remain untouched.
"""
import argparse,csv,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from dotenv import load_dotenv
from agentreel.runtime import ROOT,Run,sha,now
from agentreel.models import Brief,Script
from agentreel.pipeline import export
from agentreel import qa
from agents import editor

def main():
    load_dotenv(ROOT/'.env')
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',type=Path,required=True); p.add_argument('--media-manifest',type=Path)
    p.add_argument('--visual-template', choices=['storyboard','rainbow_optics'])
    args=p.parse_args(); parent=args.run_dir.resolve()
    brief=Brief.model_validate(yaml.safe_load((parent/'brief.yaml').read_text(encoding='utf-8')))
    if brief.sensitive: raise ValueError('Use normal generation and mandatory review for sensitive content.')
    state=json.loads((parent/'state.json').read_text())
    if state['status']!='complete': raise ValueError('Source run must have completed editorial and technical QA.')
    records=[json.loads(line) for line in (parent/'logs/run_log.jsonl').read_text(encoding='utf-8').splitlines()]
    for name in ('script.json','facts.json','storyboard.json'):
        written=[r for r in records if r.get('current_path')==name]
        if written and sha(parent/name)!=written[-1]['output_sha256']: raise ValueError('Source editorial input changed: '+name)
    ctx=Run(seed=state.get('seed',42)); print('New motion preview:',ctx.path,flush=True)
    ctx.artifact('parent_run.json',{'path':str(parent),'purpose':'motion renderer comparison; source narration unchanged','source_video_sha256':state.get('final_sha256')},'editor')
    (ctx.path/'assets').mkdir()
    for name in ('brief.yaml','facts.json','evidence.json','script.json','storyboard.json','plan.json'):
        shutil.copy2(parent/name,ctx.path/name)
    if args.visual_template:
        brief.visual_template = args.visual_template
        (ctx.path/'brief.yaml').write_text(yaml.safe_dump(brief.model_dump(),sort_keys=False,allow_unicode=True),encoding='utf-8')
    for source in parent.glob('editorial_review_*.json'): shutil.copy2(source,ctx.path/source.name)
    timing=json.loads((parent/'timings.json').read_text())
    for segment in timing['segments']:
        source=Path(segment['voice_path']); target=ctx.path/'assets'/source.name; shutil.copy2(source,target); segment['voice_path']=str(target)
    for name in ('end_silence.wav','music.wav','manifest.csv'):
        source=parent/'assets'/name
        if source.exists(): shutil.copy2(source,ctx.path/'assets'/name)
    manifest=ctx.path/'assets/manifest.csv'
    if manifest.exists():
        rows=list(csv.DictReader(manifest.open(encoding='utf-8')))
        rows=[row for row in rows if not (args.media_manifest and row.get('kind')=='stock') and ((ctx.path/row['file']).is_file() or (ROOT/row['file']).is_file())]
        with manifest.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['file','source','licence','kind','sha256','beat_index','description','start_s','identifiable_people','consent'],extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    ctx.artifact('timings.json',timing,'asset_generator')
    script=Script.model_validate_json((ctx.path/'script.json').read_text(encoding='utf-8'))
    music=ctx.path/'assets/music.wav'; music=music if music.exists() else None
    media=[]
    if args.media_manifest:
        mf=args.media_manifest.resolve()
        for row in csv.DictReader(mf.open(encoding='utf-8-sig')):
            if not row.get('source') or not row.get('licence'): raise ValueError('Media source and licence are required.')
            if brief.mode=='assignment' and row.get('kind') not in ('official','self_shot','generated'): raise ValueError('Assignment footage must be approved official/self-shot material.')
            if row.get('identifiable_people','false').lower() not in ('true','false'): raise ValueError('Review the clip and set identifiable_people to true or false.')
            if row.get('identifiable_people','false').lower()=='true' and not (row.get('consent') and (mf.parent/row['consent']).is_file()): raise ValueError('Identifiable people require consent.')
            source=(mf.parent/row['file']).resolve()
            destination=ctx.path/'assets'/source.name; shutil.copy2(source,destination)
            media.append({**row,'path':str(destination)})
        with manifest.open('a',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['file','source','licence','kind','sha256','beat_index','description','start_s','identifiable_people','consent'],extrasaction='ignore')
            for row in media: writer.writerow({**row,'file':'assets/'+Path(row['path']).name,'sha256':sha(row['path'])})
    draft=editor.run(brief,script,ctx,timing['segments'],ctx.path/'assets/end_silence.wav',[],music,media)
    result=qa.run(ctx,brief,draft)
    if not result['passed']: raise RuntimeError(str(result['checks']))
    final=export(ctx,brief,{'created_at':now(),'seed':ctx.seed,'requires_human_review':False,'render_revision':'continuous_scene_v2'})
    print('COMPLETE:',final,flush=True)
if __name__=='__main__': main()
