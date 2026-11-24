import asyncio,json,os,shutil,subprocess,wave
from pathlib import Path
import numpy as np
import soundfile as sf
from .runtime import ROOT,dump

def executable(name):
    configured=os.getenv(name.upper()+'_PATH','')
    found=configured or shutil.which(name)
    if not found: raise RuntimeError(f'{name} not found. Install FFmpeg and ensure ffmpeg and ffprobe are on PATH.')
    return found
def probe(ctx,path):
    r=ctx.command('critic',[executable('ffprobe'),'-v','error','-show_format','-show_streams','-of','json',path])
    return json.loads(r.stdout)

_kokoro=None
def local_voice(text,path,voice,language):
    global _kokoro
    from kokoro_onnx import Kokoro
    if _kokoro is None:
        model=ROOT/'.cache/kokoro/model.onnx'; voices=ROOT/'.cache/kokoro/voices.bin'
        if not model.exists() or not voices.exists(): raise FileNotFoundError('Run python scripts/setup_models.py for free local narration.')
        # Keep CPU inference responsive while FFmpeg and the CLI run.
        import onnxruntime as ort
        options=ort.SessionOptions(); options.intra_op_num_threads=4
        session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
        _kokoro=Kokoro.from_session(session,str(voices))
    audio,sr=_kokoro.create(text,voice=voice,speed=1.0,lang=language)
    sf.write(path,audio,sr,subtype='PCM_16')
    return len(audio)/sr

async def edge_voice(text,path,voice):
    import edge_tts
    communicate=edge_tts.Communicate(text,voice,rate=os.getenv('TTS_RATE','+0%'),boundary='WordBoundary')
    timings=[]
    with Path(path).open('wb') as f:
        async for chunk in communicate.stream():
            if chunk['type']=='audio': f.write(chunk['data'])
            elif chunk['type']=='WordBoundary': timings.append(chunk)
    dump(Path(path).with_suffix('.words.json'),timings)

class NarrationLengthError(ValueError):
    def __init__(self,message,target_words=None):
        super().__init__(message); self.target_words=target_words

def pacing(lengths,body_seconds):
    """Preserve natural delivery and leave bounded pauses for visual actions."""
    available=body_seconds-.18*len(lengths)
    ratio=sum(lengths)/available
    tempo=max(.9,min(1.12,ratio))
    gap=(body_seconds-sum(lengths)/tempo)/len(lengths)
    return tempo,gap,ratio

def voice_settings(brief):
    lang=brief.language.lower()
    backend=os.getenv('TTS_BACKEND','kokoro')
    if 'marathi' in lang: backend='edge'
    if backend=='kokoro':
        language='hi' if 'hindi' in lang else 'en-us'
        voice=brief.voice or ('hf_alpha' if language=='hi' else os.getenv('KOKORO_VOICE','af_heart'))
    else:
        language=lang; voice=brief.voice or ('mr-IN-AarohiNeural' if 'marathi' in lang else 'hi-IN-SwaraNeural' if 'hindi' in lang else os.getenv('TTS_VOICE','en-IN-NeerjaNeural'))
    return backend,language,voice

def speak(ctx,brief,script):
    """Raw narration per beat. Identical text reuses earlier audio, so measuring a draft
    script during writing costs nothing again at asset time."""
    import hashlib
    folder=ctx.path/'assets'; folder.mkdir(exist_ok=True)
    backend,language,voice=voice_settings(brief)
    raw=[]; lengths=[]
    for i,beat in enumerate(script.beats):
        key=hashlib.sha256(f'{backend}|{voice}|{language}|{beat.narration}'.encode()).hexdigest()[:16]
        path=folder/f'voice_{key}.{"wav" if backend=="kokoro" else "mp3"}'
        if path.is_file():
            duration=float(probe(ctx,path)['format']['duration'])
        elif backend=='kokoro':
            duration=ctx.tool('asset_generator','kokoro_tts',lambda:local_voice(beat.narration,path,voice,language),{'text':beat.narration,'voice':voice,'language':language})
        else:
            ctx.tool('asset_generator','edge_tts',lambda:asyncio.run(edge_voice(beat.narration,path,voice)),{'text':beat.narration,'voice':voice})
            duration=float(probe(ctx,path)['format']['duration'])
        raw.append(path); lengths.append(duration)
    return raw,lengths,backend,voice

def check_length(brief,script,lengths):
    """Raise with a word target computed from this voice's measured speaking rate."""
    body=brief.duration_s-4
    tempo,gap,ratio=pacing(lengths,body)
    if .12<=gap<=1.5: return tempo,gap
    words=sum(len(b.narration.split()) for b in script.beats)
    # Aim for natural speed with ~0.45 s breathing room per beat.
    target=max(20,round(words*(body-.45*len(lengths))/sum(lengths)))
    raise NarrationLengthError(f'Narration is {sum(lengths):.1f}s for {body:.1f}s available. Rewrite narration to approximately {target} words in total (currently {words}), retaining all verified facts and requirements.',target)

def synthesize(ctx,brief,script):
    folder=ctx.path/'assets'
    raw,lengths,backend,voice=speak(ctx,brief,script)
    tempo,gap=check_length(brief,script,lengths)
    segments=[]; clock=0
    for i,(path,length) in enumerate(zip(raw,lengths)):
        duration=length/tempo+gap
        duration=round(duration*brief.fps)/brief.fps
        if i==len(raw)-1: duration=brief.duration_s-4-clock
        dest=folder/f'voice_fit_{i:02}.wav'
        ctx.command('asset_generator',[executable('ffmpeg'),'-y','-v','error','-i',path,'-af',f'atempo={tempo:.8f},apad','-t',f'{duration:.8f}','-ar','48000','-ac','2','-c:a','pcm_s16le',dest])
        segments.append({'index':i,'start':clock,'end':clock+duration,'duration':duration,'voice_path':str(dest),'spoken_end':clock+length/tempo})
        clock+=duration
    end=folder/'end_silence.wav'; sf.write(end,np.zeros((4*48000,2),dtype=np.float32),48000,subtype='PCM_16')
    ctx.artifact('timings.json',{'segments':segments,'end_card_s':4,'voice':voice,'backend':backend,'tempo':tempo,'subtitle_alignment':'beat-level boundaries from actual generated audio, not forced word alignment'},'asset_generator')
    return segments,end

def make_music(ctx,duration):
    def create():
        sr=48000; t=np.arange(int(sr*duration),dtype=np.float32)/sr
        audio=np.zeros_like(t)
        # Original ambient motif, generated mathematically; no sampled music.
        for i,f in enumerate((130.8128,164.8138,195.9977,261.6256)):
            swell=.5+.5*np.sin(2*np.pi*.075*t+i*.7)
            audio+=.065*swell*(np.sin(2*np.pi*f*t)+.15*np.sin(2*np.pi*f*2*t))
        audio*=np.minimum(t/2,1)*np.minimum((duration-t)/3,1)
        path=ctx.path/'assets/music.wav'; sf.write(path,np.stack([audio,audio*.94],axis=1),sr,subtype='PCM_16'); return path
    return ctx.tool('asset_generator','procedural_music',create,{'duration_s':duration,'licence':'Original procedural composition generated by AgentReel; no third-party samples.'})
