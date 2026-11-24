import csv,json
from pathlib import Path
from agentreel.runtime import ROOT,sha
from agentreel.speech import synthesize,make_music
from agentreel.visuals import BRAND

def run(brief,script,board,ctx,media_manifest=None,music=True,stock_source="auto",illustrations='auto',video_model='auto'):
    timings,endvoice=synthesize(ctx,brief,script)
    # The editor renders the storyboard continuously; static PNG slides are no longer generated.
    frames=[]
    ctx.emit('asset_generator','prepare_continuous_scenes',inputs={'seed':ctx.seed,'brand':brief.brand,'shots':len(board.shots)})
    soundtrack=make_music(ctx,brief.duration_s) if music else None
    media=[]; manifest=[]
    if media_manifest:
        mf=Path(media_manifest).resolve()
        for row in csv.DictReader(mf.open(encoding='utf-8-sig')):
            if not row.get('source') or not row.get('licence'): raise ValueError('Every media asset needs source and licence.')
            path=(mf.parent/row['file']).resolve()
            if path.suffix.lower() not in ('.png','.jpg','.jpeg','.webp','.mp4','.mov','.mkv','.webm','.ogv'): raise ValueError('Unsupported media asset '+str(path))
            if not path.is_file(): raise FileNotFoundError(path)
            if row.get('identifiable_people','false').lower() not in ('true','false'): raise ValueError('Review the clip and set identifiable_people to true or false in the media CSV.')
            if row.get('identifiable_people','false').lower()=='true' and not (row.get('consent') and (mf.parent/row['consent']).is_file()): raise ValueError('Identifiable footage requires a consent file.')
            if brief.mode=='assignment' and row.get('kind') not in ('official','self_shot','generated'): raise ValueError('Assignment media must be official, self_shot with consent, or generated.')
            assigned=row.get('beat_index','')
            if assigned and not 0<=int(assigned)<len(script.beats): raise ValueError('Media beat_index is outside the script.')
            if float(row.get('start_s') or 0)<0: raise ValueError('Media start_s must be nonnegative.')
            media.append({**row,'path':str(path)}); manifest.append({**row,'file':str(path),'sha256':sha(path)})
    from agentreel.stock import collect
    stock=collect(brief,board,ctx,media,stock_source,{t['index']:t['end']-t['start'] for t in timings})
    media.extend(stock); manifest.extend(stock)
    from agentreel.illustrate import collect as illustrate
    painted,extra=illustrate(brief,board,ctx,media,illustrations)
    # Where a large enough GPU exists, painted stills become real moving clips.
    from agentreel.animate import collect as animate
    extra+=animate(brief,board,ctx,painted,{t['index']:t['end']-t['start'] for t in timings},video_model)
    media.extend(painted); manifest.extend(painted+extra)
    for b,o in frames:
        for p in (b,o): manifest.append({'file':str(p.relative_to(ctx.path)),'source':'AgentReel procedural graphics','licence':'Original code-generated artwork; institutional marks retain their owners rights','kind':'generated','sha256':sha(p)})
    if brief.brand=='somaiya':
        for role,name in BRAND.items():
            p=ROOT/'assets/brand'/name
            manifest.append({'file':str(p.relative_to(ROOT)),'source':'User-supplied somaiya logos.odt; '+role,'licence':'Institutional branding supplied for this project; no transfer of trademark rights','kind':'official','sha256':sha(p)})
    for p in (ctx.path/'assets').glob('*.wav'):
        manifest.append({'file':str(p.relative_to(ctx.path)),'source':'Kokoro/Edge neural speech or original procedural audio','licence':'See THIRD_PARTY.md; no cloned real-person voice','kind':'generated','sha256':sha(p)})
    with (ctx.path/'assets/manifest.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['file','source','licence','kind','sha256','beat_index','description','start_s','identifiable_people','consent'],extrasaction='ignore'); writer.writeheader(); writer.writerows(manifest)
    return timings,endvoice,frames,soundtrack,media
