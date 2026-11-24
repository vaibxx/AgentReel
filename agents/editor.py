import json,math,re
from pathlib import Path
from agentreel.speech import executable
from agentreel.runtime import dump

def stamp(seconds):
    ms=round(seconds*1000); h,ms=divmod(ms,3600000); m,ms=divmod(ms,60000); s,ms=divmod(ms,1000)
    return f'{h:02}:{m:02}:{s:02},{ms:03}'
def srt(ctx,script,timings):
    text='\n\n'.join(f'{i+1}\n{stamp(t["start"])} --> {stamp(t["end"])}\n{b.caption}' for i,(b,t) in enumerate(zip(script.beats,timings)))+'\n'
    (ctx.path/'captions.srt').write_text(text,encoding='utf-8')
    (ctx.path/'script.md').write_text('# Timed script\n\n'+'\n\n'.join(f'## {stamp(t["start"])} – {stamp(t["end"])}\n{b.narration}\n\nEvidence: {", ".join(b.fact_ids)}' for b,t in zip(script.beats,timings))+'\n\nCTA: '+script.cta,encoding='utf-8')

def mix_filter(music,tail):
    return ('[1:a][2:a]amix=inputs=2:duration=first:weights=\'1 0.09\':normalize=0,' if music else '[1:a]')+tail

def run(brief,script,ctx,timings,endvoice,frames,music,media):
    from agentreel.motion import render,caption_cues,render_script
    from agentreel.models import Storyboard
    ff=executable('ffmpeg')
    script=render_script(brief,script)
    ctx.artifact('render_script.json',script.model_dump(),'editor')
    board=Storyboard.model_validate_json((ctx.path/'storyboard.json').read_text(encoding='utf-8'))
    silent=render(ctx,brief,script,board,timings,media)
    srt(ctx,script,timings)
    cues=caption_cues(script,timings)
    (ctx.path/'captions.srt').write_text('\n\n'.join(f'{i+1}\n{stamp(c["start"])} --> {stamp(c["end"])}\n{c["text"]}' for i,c in enumerate(cues))+'\n',encoding='utf-8')
    ctx.artifact('caption_cues.json',cues,'editor')
    concat=ctx.path/'audio_concat.txt'
    voice_paths=[Path(t['voice_path']) for t in timings]+[endvoice]
    concat.write_text('\n'.join("file 'assets/"+p.name+"'" for p in voice_paths),encoding='utf-8')
    inputs=[ff,'-y','-hide_banner','-i',str(silent),'-f','concat','-safe','0','-i',str(concat)]
    if music: inputs+=['-i',str(music)]
    measure=ctx.command('editor',inputs+['-filter_complex',mix_filter(music,'loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json[a]'),'-map','[a]','-vn','-f','null','-'])
    matches=re.findall(r'\{\s*"input_i".*?\}',measure.stderr,re.S)
    if not matches: raise RuntimeError('FFmpeg did not return loudness measurements.')
    measured=json.loads(matches[-1]); ctx.artifact('audio_pre_normalization.json',measured,'editor')
    norm=f'loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={measured["input_i"]}:measured_TP={measured["input_tp"]}:measured_LRA={measured["input_lra"]}:measured_thresh={measured["input_thresh"]}:offset={measured["target_offset"]}:linear=true[a]'
    draft=ctx.path/'draft.mp4'
    ctx.command('editor',inputs+['-filter_complex',mix_filter(music,norm),'-map','0:v:0','-map','[a]','-c:v','copy','-c:a','aac','-b:a','160k','-ar','48000','-ac','2','-t',str(brief.duration_s),'-movflags','+faststart',draft])
    return draft
