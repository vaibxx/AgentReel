import json,re
from fractions import Fraction
from .speech import executable,probe
from .runtime import sha,verify_log

def run(ctx,brief,draft):
    info=probe(ctx,draft); v=next(s for s in info['streams'] if s['codec_type']=='video'); a=next(s for s in info['streams'] if s['codec_type']=='audio')
    duration=float(info['format']['duration']); w,h=brief.size
    measurement=ctx.command('critic',[executable('ffmpeg'),'-hide_banner','-i',draft,'-vn','-af','loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json','-f','null','-'])
    loud=json.loads(re.findall(r'\{\s*"input_i".*?\}',measurement.stderr,re.S)[-1])
    layout=json.loads((ctx.path/'layout.json').read_text(encoding='utf-8')); timings=json.loads((ctx.path/'timings.json').read_text(encoding='utf-8'))
    cues=re.findall(r'(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})',(ctx.path/'captions.srt').read_text(encoding='utf-8'))
    bounds=[(int(c[0])*3600+int(c[1])*60+int(c[2])+int(c[3])/1000,int(c[4])*3600+int(c[5])*60+int(c[6])+int(c[7])/1000) for c in cues]
    checks={
      'duration_30_to_60':30<=duration<=60.10,
      'requested_duration':abs(duration-brief.duration_s)<=.15,
      'resolution':(v['width'],v['height'])==(w,h),
      'frame_rate':abs(float(Fraction(v['avg_frame_rate']))-brief.fps)<.01 and v['r_frame_rate']==v['avg_frame_rate'],
      'h264_high':v['codec_name']=='h264' and v.get('profile')=='High',
      'pixel_format':v['pix_fmt']=='yuv420p',
      'aac_stereo_48khz':a['codec_name']=='aac' and int(a['sample_rate'])==48000 and a['channels']==2,
      'audio_bitrate':128000<=int(a.get('bit_rate',160000))<=192000,
      'file_size':draft.stat().st_size<=100_000_000,
      'loudness':abs(float(loud['input_i'])+14)<=1.5,
      'no_clipping':float(loud['input_tp'])<=-.1,
      'subtitles_file':(ctx.path/'captions.srt').stat().st_size>30,
      'subtitle_timing':bool(bounds) and len(bounds)>=len(timings['segments']) and all(0<=s<e<=duration for s,e in bounds) and all(bounds[i][1]<=bounds[i+1][0] for i in range(len(bounds)-1)),
      'frame_count':abs(int(v.get('nb_frames',0))-brief.duration_s*brief.fps)<=1,
      'subtitle_font':all(box['font_size']>=48 for item in layout[:-1] for box in item['text'] if box.get('role')=='caption') if (ctx.path/'motion_plan.json').exists() else all(item['text'][-1]['font_size']>=48 if brief.brand=='none' else item['text'][-2]['font_size']>=48 for item in layout[:-1]),
      'end_card':timings['end_card_s']==4,
      'branding':all(item['branding']==brief.brand for item in layout),
      'safe_zones':all(box['box'][1]>=250 and box['box'][1]+box['box'][3]<=h-350 for item in layout for box in item['text']) if h>w else True,
      'log_integrity':verify_log(ctx.logfile)>0,
    }
    report={'passed':all(checks.values()),'checks':checks,'duration_s':duration,'bytes':draft.stat().st_size,'audio':loud,'draft_sha256':sha(draft),'branding':brief.brand,'requires_human_review':brief.sensitive,'notes':['Captions are burned into the video and supplied as matching SRT cues.','Phrase alignment is estimated within measured beats; not word-level forced alignment.','Factual verification is in editorial review files and source-linked facts.json.','Automated layout checks do not replace visual inspection.']}
    ctx.artifact('qa_report.json',report,'critic'); return report

def repair(ctx,brief,draft,report,attempt):
    """Repair encoding/loudness only; never hide factual/layout failures."""
    failed={k for k,v in report['checks'].items() if not v}
    repairable={'file_size','loudness','no_clipping','audio_bitrate','aac_stereo_48khz','h264_high','pixel_format','frame_rate','frame_count'}
    if not failed or not failed<=repairable: return False
    target=draft.with_name(f'technical_repair_{attempt}.mp4')
    ctx.command('editor',[executable('ffmpeg'),'-y','-v','error','-i',draft,'-c:v','libx264','-profile:v','high','-pix_fmt','yuv420p','-r',str(brief.fps),'-fps_mode','cfr','-crf','21','-maxrate','7M','-bufsize','14M','-threads','4','-af','loudnorm=I=-14:TP=-1.5:LRA=11','-c:a','aac','-b:a','160k','-ar','48000','-ac','2','-movflags','+faststart',target])
    draft.replace(draft.with_name(f'rejected_draft_{attempt}.mp4'))
    target.replace(draft)
    return True
