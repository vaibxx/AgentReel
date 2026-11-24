import json,shutil,time
from pathlib import Path
import yaml
from .runtime import ROOT,Run,dump,sha,now
from .provider import Provider
from .models import Brief,Script,Plan
from .speech import NarrationLengthError
from . import qa
from .reports import report
from agents import planner,researcher,scriptwriter,storyboarder,asset_generator,editor,critic

def review_hash(ctx):
    return {name:sha(ctx.path/name) for name in ('brief.yaml','facts.json','script.json','storyboard.json','draft.mp4','qa_report.json')}

def export(ctx,brief,state):
    ctx.check()
    name=f'{brief.team_id}_{brief.id}_final.mp4'
    target=ctx.path/name; shutil.copy2(ctx.path/'draft.mp4',target)
    shutil.copy2(ctx.path/'captions.srt',ctx.path/f'{brief.team_id}_{brief.id}_final.srt')
    state.update(status='complete',final_file=name,final_sha256=sha(target),completed_at=now())
    ctx.artifact('state.json',state,'editor'); report(ctx,brief,'complete')
    ctx.emit('editor','export',output_path=name,output_sha256=sha(target))
    return target

def generate(args):
    ctx=Run(seed=args.seed); provider=Provider(ctx)
    print('Run directory:',ctx.path,flush=True)
    if args.brief:
        brief=Brief.model_validate(yaml.safe_load(Path(args.brief).read_text(encoding='utf-8')))
        ctx.emit('parser','read_yaml',inputs={'path':str(args.brief)})
    else: brief=provider.generate('parser',Brief,{'request':args.prompt})
    if args.team_id: brief.team_id=args.team_id
    if args.duration: brief.duration_s=args.duration
    if args.format: brief.format=args.format
    if args.brand: brief.brand=args.brand
    if args.mode: brief.mode=args.mode
    brief=Brief.model_validate(brief.model_dump())
    (ctx.path/'brief.yaml').write_text(yaml.safe_dump(brief.model_dump(),sort_keys=False,allow_unicode=True),encoding='utf-8')
    state={'status':'running','created_at':now(),'seed':args.seed,'live':args.live,'requires_human_review':brief.sensitive,'brief_id':brief.id}
    ctx.artifact('state.json',state,'planner')
    try:
        print('Planner → Researcher',flush=True)
        plan=planner.run(brief,provider,ctx)
        facts=researcher.run(brief,plan,provider,ctx,args.documents or [])
        if facts['gaps'] and brief.must_include:
            # The critic must resolve all required gaps; retain them visibly.
            print('Research gaps recorded; editorial review will enforce required inclusions.',flush=True)
        feedback=''; script=None; board=None; target=None
        from .imagegen import order
        # Offer illustration mode only when some image backend (API, Space or local GPU) can paint.
        backends=order() if getattr(args,'illustrations','auto')!='none' else []
        paint=bool(backends)
        for loop in range(3):
            print(f'Script → Storyboard → Editorial QA (attempt {loop+1}/3)',flush=True)
            script=scriptwriter.run(brief,facts,provider,ctx,feedback,target)
            board=storyboarder.run(brief,script,provider,ctx,paint,plentiful=any(b in ('cloudflare','hf') for b in backends))
            review=critic.editorial(brief,facts,script,provider,ctx,board)
            if not review.passed:
                feedback='; '.join(review.issues+review.unsupported_claims); continue
            try:
                print('Generating neural narration and visual assets',flush=True)
                assets=asset_generator.run(brief,script,board,ctx,args.media_manifest,not args.no_music,getattr(args,"stock_source","auto"),getattr(args,'illustrations','auto'),getattr(args,'video_model',None))
                break
            except NarrationLengthError as exc: feedback=str(exc); target=exc.target_words
        else: raise RuntimeError('Critic/repair loop exhausted. '+feedback)
        print('Rendering video and running technical QA',flush=True)
        draft=editor.run(brief,script,ctx,*assets)
        checked=qa.run(ctx,brief,draft)
        for repair_attempt in range(1,3):
            if checked['passed'] or not qa.repair(ctx,brief,draft,checked,repair_attempt): break
            checked=qa.run(ctx,brief,draft)
        if not checked['passed']: raise RuntimeError('Technical QA failed: '+', '.join(k for k,v in checked['checks'].items() if not v))
        if brief.sensitive:
            state.update(status='awaiting_human_review',review_files=review_hash(ctx))
            ctx.artifact('state.json',state,'critic'); report(ctx,brief,'awaiting mandatory human review')
            print('Review draft.mp4, script.json, facts.json and qa_report.json, then run:\npython run.py --approve "'+str(ctx.path)+'" --reviewer "Your name"',flush=True)
            return ctx.path
        final=export(ctx,brief,state); print('COMPLETE:',final,flush=True); return ctx.path
    except Exception as exc:
        state.update(status='failed',error_type=type(exc).__name__,message=str(exc)[:1500])
        ctx.artifact('state.json',state,'orchestrator')
        print('Run stopped safely. Logs and intermediate files:',ctx.path,flush=True)
        raise

def approve(path,reviewer):
    ctx=Run(path); state=json.loads((ctx.path/'state.json').read_text(encoding='utf-8'))
    if state['status']!='awaiting_human_review': raise ValueError('Run is not awaiting review.')
    if not reviewer.strip(): raise ValueError('A reviewer name is required.')
    if review_hash(ctx)!=state['review_files']: raise ValueError('Reviewed files changed. Re-run QA and generate a new review bundle.')
    if state.get('live'):
        from datetime import datetime,timezone
        if (datetime.now(timezone.utc)-datetime.fromisoformat(state['created_at'])).total_seconds()>10800: raise ValueError('Live run exceeded the three-hour deadline.')
    brief=Brief.model_validate(yaml.safe_load((ctx.path/'brief.yaml').read_text(encoding='utf-8')))
    ctx.artifact('human_review.json',{'reviewer':reviewer,'approved_at':now(),'reviewed_files':state['review_files'],'counts_as_intervention':False},'human_review')
    state['human_review_approved']=True
    return export(ctx,brief,state)

def rerun_stage(path,stage,seed):
    parent=Path(path).resolve(); ctx=Run(seed=seed); provider=Provider(ctx)
    brief=Brief.model_validate(yaml.safe_load((parent/'brief.yaml').read_text(encoding='utf-8')))
    ctx.artifact('parent_run.json',{'path':str(parent),'stage':stage},'orchestrator')
    if stage=='planner': planner.run(brief,provider,ctx)
    else:
        facts=json.loads((parent/'facts.json').read_text(encoding='utf-8'))
        if stage=='scriptwriter': scriptwriter.run(brief,facts,provider,ctx)
        else:
            script=Script.model_validate_json((parent/'script.json').read_text(encoding='utf-8'))
            if stage=='storyboarder': storyboarder.run(brief,script,provider,ctx)
            elif stage=='critic': critic.editorial(brief,facts,script,provider,ctx)
    print('Stage re-run saved separately:',ctx.path)
    return ctx.path
