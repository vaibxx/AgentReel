"""Continuous scene compositor. Objects evolve over time; no PNG slide zooms.

Illustrative simulations are deliberately not represented as real campus footage.
Voice remains on one uninterrupted timeline. Crossfades affect scene imagery only.
"""
import bisect,csv,json,math,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageEnhance
from .runtime import ROOT,sha
from .speech import executable
from .visuals import BRAND,ACCENTS,font,textblock,fit_image

def ease(x):
    x=max(0.,min(1.,x)); return x*x*(3-2*x)

def render_script(brief,script):
    result=script.model_copy(deep=True)
    if brief.language.strip().casefold()==brief.subtitle_language.strip().casefold():
        for beat in result.beats: beat.caption=beat.narration
    return result

def caption_cues(script,timings):
    """Short phrase captions, estimated within measured beat audio boundaries."""
    cues=[]
    for beat,timing in zip(script.beats,timings):
        groups=[]; words=[]
        for word in beat.caption.split():
            if words and (len(words)>=6 or len(' '.join(words+[word]))>42):
                groups.append(' '.join(words)); words=[]
            words.append(word)
        if words: groups.append(' '.join(words))
        weights=[max(1,len(s)) for s in groups]; total=sum(weights); at=timing['start']
        spoken_end=min(timing['end'],timing.get('spoken_end',timing['end']))
        for j,(group,weight) in enumerate(zip(groups,weights)):
            end=at+(spoken_end-timing['start'])*weight/total
            if j==len(groups)-1: end=timing['end']
            cues.append({'start':at,'end':end,'text':group,'beat_index':timing['index']}); at=end
    return cues

def demonstration(brief,beat,shot):
    if shot.demonstration!='auto': return shot.demonstration
    text=(brief.topic+' '+beat.narration+' '+shot.visual_description).lower()
    if any(w in text for w in ('rainbow','refraction','droplet','dispersion')): return 'optics'
    local=(beat.narration+' '+shot.label).lower()
    if any(w in local for w in ('mechanical','gear','motor','machine')): return 'gears'
    if any(w in local for w in ('computer','artificial intelligence','data','circuit','electronic')): return 'signal_flow'
    if any(w in local for w in ('programme','program','credit','degree','career','choice','learn')): return 'pathway'
    return {'orbit':'assembly','signal':'signal_flow','layers':'assembly','path':'pathway','network':'network','grid':'network'}[shot.visual]

def media_for_beat(media,index,beat):
    """Explicit mapping wins. Legacy untagged rows map once in order, never cycle."""
    terms=set((beat.narration+' '+beat.headline).lower().split())
    candidates=[]
    for position,item in enumerate(media):
        if isinstance(item,(str,Path)): item={'path':str(item),'beat_index':position}
        assigned=item.get('beat_index')
        if assigned not in (None,''):
            if int(assigned)==index: return item
            continue
        tags=set(item.get('description','').lower().split())
        if tags: candidates.append((len(terms&tags),item))
        elif position==index: candidates.append((1,item))
    return max(candidates,key=lambda x:x[0])[1] if candidates and max(x[0] for x in candidates)>0 else None

class Footage:
    def __init__(self,item,size,fps,duration=None):
        self.item=item; self.path=Path(item['path']); self.proc=None; self.size=size
        w,h=size
        if self.path.suffix.lower() in ('.mp4','.mov','.mkv','.webm','.ogv'):
            # Overscan, then a slow eased camera drift that alternates direction between
            # shots, plus a gentle grade and vignette so mixed sources feel like one film.
            sw,sh=math.ceil(w*1.1/2)*2,math.ceil(h*1.1/2)*2
            sign=-1 if int(item.get('beat_index') or 0)%2 else 1
            drift=f'(0.5+{sign}*0.5*sin(min(t/9,1)*PI/2))'
            fit=''; loop=['-stream_loop','-1']
            if item.get('clip_s') and duration:
                # Generated clips are short: slow them gently to fill the beat, then hold the
                # last frame, instead of looping back to the start mid-shot.
                stretch=max(1.,min(1.6,duration/max(.1,float(item['clip_s']))))
                fit=f'setpts={stretch:.4f}*PTS,tpad=stop_mode=clone:stop_duration={duration+1:.2f},'; loop=[]
            vf=(f'{fit}fps={fps},scale={sw}:{sh}:force_original_aspect_ratio=increase,'
                f"crop={w}:{h}:x='(iw-ow)*{drift}':y='(ih-oh)*(0.5-0.25*{sign}*min(t/9,1))',"
                'eq=contrast=1.06:saturation=1.08:brightness=-0.015,vignette=angle=PI/5')
            args=[executable('ffmpeg'),'-v','error','-ss',str(float(item.get('start_s') or 0)),*loop,'-i',str(self.path),'-an','-vf',vf,'-threads','2','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
            self.proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        else:
            from PIL import ImageOps
            self.still=ImageOps.fit(Image.open(self.path).convert('RGB'),size,method=Image.Resampling.LANCZOS)
    def frame(self,t):
        if self.proc:
            data=self.proc.stdout.read(self.size[0]*self.size[1]*3)
            if len(data)!=self.size[0]*self.size[1]*3: raise RuntimeError('Footage decoder ended unexpectedly: '+self.path.name)
            return Image.frombytes('RGB',self.size,data)
        # Stills remain an explicit supplied-media fallback, not the default renderer.
        w,h=self.size; zoom=1+.035*min(t/5,1); im=self.still.resize((round(w*zoom),round(h*zoom)),Image.Resampling.BICUBIC)
        x=(im.width-w)//2; y=(im.height-h)//2
        return im.crop((x,y,x+w,y+h))
    def close(self):
        if self.proc:
            self.proc.stdout.close(); self.proc.terminate(); self.proc.wait(timeout=10)

class Sequence:
    """One AI-generated beat played like edited footage: each still gets its own camera
    move (depth parallax), generated clips play at their natural pace, parts are joined by
    short crossfades, and the scene's text objects stay on top for the whole beat."""
    CAMERAS=('push_in','track_left','track_right','orbit')
    FADE=.3
    def __init__(self,record,size,fps,camera,duration,overlay=None):
        from .illustration import Illustration
        if record.get('parts'): parts=record['parts']
        elif record.get('video'): parts=[{'video':record['path'],'clip_s':record.get('clip_s')}]
        else: parts=[{'path':record['path'],'depth':record.get('depth','')}]
        self.overlay=overlay; self.size=size; beat=int(record.get('beat_index') or 0)
        self.durations=self.split(parts,duration); self.kinds=[]; self.renderers=[]
        cameras=[camera]+[c for c in self.CAMERAS if c!=camera]
        for i,(part,length) in enumerate(zip(parts,self.durations)):
            if part.get('video'):
                item={'path':part['video'],'clip_s':part.get('clip_s'),'beat_index':beat+i,'ai_generated':True,'kind':'generated'}
                self.renderers.append(Footage(item,size,fps,length)); self.kinds.append('video')
            else:
                item={'path':part['path'],'depth':part.get('depth',''),'beat_index':beat+i}
                self.renderers.append(Illustration(item,size,cameras[i%len(cameras)],length)); self.kinds.append('image')
        self.starts=[sum(self.durations[:i]) for i in range(len(parts))]
    @staticmethod
    def split(parts,duration):
        if len(parts)==1: return [duration]
        # A clip keeps its natural pace (at most 1.6x slower) but never takes over the beat.
        want={i:min(duration*.6,max(duration/len(parts),float(p.get('clip_s') or 0)*1.6)) for i,p in enumerate(parts) if p.get('video')}
        stills=[i for i in range(len(parts)) if i not in want]
        if not stills: return [duration/len(parts)]*len(parts)
        rest=max(.5,(duration-sum(want.values()))/len(stills))
        return [want.get(i,rest) for i in range(len(parts))]
    def index(self,t):
        return max(0,bisect.bisect_right(self.starts,t)-1)
    def label(self,t):
        return 'AI-GENERATED VIDEO' if self.kinds[self.index(t)]=='video' else 'AI-GENERATED IMAGE'
    def frame(self,t):
        i=self.index(t); local=t-self.starts[i]
        im=self.renderers[i].frame(local)
        if i and local<self.FADE:
            previous=self.renderers[i-1].frame(self.durations[i-1]+local)
            im=Image.blend(previous,im,ease(local/self.FADE))
        if self.overlay is not None:
            im=im.convert('RGBA'); im.alpha_composite(self.overlay.frame(t)); im=im.convert('RGB')
        return im
    def close(self):
        for r in self.renderers:
            if isinstance(r,Footage): r.close()

class Scene:
    def __init__(self,brief,beat,shot,index,size,media=None,duration=4,theme='midnight'):
        self.brief=brief; self.beat=beat; self.shot=shot; self.index=index; self.size=size
        self.kind=demonstration(brief,beat,shot); self.color=ACCENTS[shot.accent]
        self.seed=0
        words=(beat.narration+' '+shot.label+' '+shot.visual_description).lower()
        self.optic_phase='circle' if 'full circle' in words else 'spectrum' if any(q in words for q in ('dispers','separat','color spectrum')) else 'ray' if any(q in words for q in ('refract','reflect','enters','droplet','exiting')) else 'sky'
        self.illustration=None; self.overlay=None
        if media and media.get('ai_generated'):
            # AI stills and generated clips: cut together, with the scene's text objects on top.
            from .composition import Composition
            overlay=Composition(shot.scene,size,shot.camera,duration,theme,overlay=True) if shot.scene is not None else None
            self.illustration=Sequence(media,size,brief.fps,shot.camera,duration,overlay)
            media=None
        self.footage=Footage(media,size,brief.fps) if media else None
        self.composition=None
        if getattr(brief, 'visual_template', 'storyboard') == 'rainbow_optics' and not media:
            from .rainbow import RainbowDiagram
            self.composition = RainbowDiagram(size, beat.narration, duration)
            self.kind = 'rainbow_optics'
        elif shot.scene is not None and not self.illustration:
            from .composition import Composition
            self.composition=Composition(shot.scene,size,shot.camera,duration,theme)
            self.kind='composition'
        w,h=size; yy,xx=np.mgrid[0:h,0:w]
        radial=np.exp(-(((xx-w*.5)/(w*.8))**2+((yy-h*.5)/(h*.55))**2))
        self.base=Image.fromarray(np.stack([np.clip(8+radial*c*.12+yy/h*5,0,255).astype('uint8') for c in (75,140,190)],axis=2))
    def frame(self,t,global_t):
        if self.illustration: return self.illustration.frame(t)
        if self.footage: return self.footage.frame(t)
        if self.composition: return self.composition.frame(t)
        im=self.base.copy(); d=ImageDraw.Draw(im); w,h=self.size
        cx=w*.5; cy=h*.52; scale=min(w,h)*.36
        # A shared vanishing point and direction keep neighboring scenes connected.
        drift=math.sin(global_t*.15)*w*.018
        for j in range(0 if self.kind=='optics' else 15):
            depth=((j/15+global_t*.035)%1)**2
            y=cy+depth*h*.55
            d.line((0,y,w,y),fill=(18+int(depth*16),30+int(depth*16),42+int(depth*16)),width=1)
        if self.kind!='optics':
            for j in range(-7,8): d.line((cx+drift,cy,w*.5+j*w*.24,h),fill=(22,34,44),width=1)
        for j in range(25):
            z=.25+(j%5)*.18; x=((j*83.7+self.seed*3.1+global_t*8*z)%(w+40))-20; y=(j*107.3+self.seed*5.7)%(h*.8)
            r=1+z*2; d.ellipse((x-r,y-r,x+r,y+r),fill=(50,70,85))
        cam=1+.045*ease(t/4)
        dx=math.sin(t*.35+self.index)*w*.025
        if self.shot.camera=='track_left': dx=-min(t,6)*w*.007
        if self.shot.camera=='track_right': dx=min(t,6)*w*.007
        if self.shot.camera=='orbit': dx=math.sin(t*.28)*w*.045
        cx+=dx; scale*=cam
        getattr(self,'draw_'+self.kind)(d,cx,cy,scale,t,global_t)
        return im
    def draw_optics(self,d,x,y,r,t,g):
        # Conceptual ray path: entry, internal reflection and dispersed exit.
        colors=[(245,80,90),(250,155,70),(245,210,90),(90,200,135),(75,160,235),(160,110,230)]
        if self.optic_phase in ('sky','circle'):
            full=self.optic_phase=='circle'; centre=y+r*.25
            for i,c in enumerate(colors):
                rr=r*(1.2-i*.035)
                end=180+(360 if full else 180)*ease(t/1.8)
                d.arc((x-rr,centre-rr,x+rr,centre+rr),180,end,fill=c,width=max(2,round(r*.025)))
            # Horizon obscures the lower arc in the establishing shot, then reveals it.
            if not full:
                w,h=self.size
                d.polygon([(0,y+r*.62),(w*.22,y+r*.3),(w*.5,y+r*.67),(w*.75,y+r*.35),(w,y+r*.6),(w,h),(0,h)],fill=(15,40,45))
                d.polygon([(0,y+r*.9),(w*.4,y+r*.65),(w*.7,y+r*.95),(w,y+r*.73),(w,h),(0,h)],fill=(12,29,35))
            for j in range(18):
                px=(j*61+g*7)%self.size[0]; py=y-r+((j*.17+g*.12)%1)*r*1.8
                d.line((px,py,px-3,py+14),fill=(75,125,145),width=2)
            return
        if self.optic_phase=='spectrum':
            x+=r*.3; r*=1.12
        d.ellipse((x-r*.7,y-r*.7,x+r*.7,y+r*.7),fill=(22,58,83),outline=(100,190,215),width=3)
        for i in range(5):
            q=.7-i*.018; d.arc((x-r*q,y-r*q,x+r*q,y+r*q),195,290,fill=(45+i*13,115+i*12,145+i*10),width=2)
        start=(x-r*1.4,y-r*.52); enter=(x-r*.55,y-r*.44); bounce=(x+r*.65,y+r*.08); leave=(x-r*.35,y+r*.6)
        d.line([start,enter,bounce,leave],fill=(225,229,210),width=3)
        for i,c in enumerate(colors):
            end=(x-r*1.35,y+r*(.68+i*.075)); d.line([leave,end],fill=c,width=3)
            f=((t*.38+i*.02)%1); px=leave[0]+(end[0]-leave[0])*f; py=leave[1]+(end[1]-leave[1])*f
            d.ellipse((px-4,py-4,px+4,py+4),fill=c)
        pts=[start,enter,bounce,leave]
        p=(t*.6)%3; k=int(p); f=p-k; a,b=pts[k],pts[k+1]; px=a[0]+(b[0]-a[0])*f; py=a[1]+(b[1]-a[1])*f
        d.ellipse((px-6,py-6,px+6,py+6),fill=(255,250,214))
        # Falling droplets in a separate depth plane make the atmosphere move.
        for j in range(12):
            px=x+math.sin(j*2.3)*r*1.4; py=y-r*1.5+((j*.21+g*.13)%1)*r*3
            d.line((px,py,px-2,py+13),fill=(55,105,140),width=2)
    def draw_gears(self,d,x,y,r,t,g):
        for i,(dx,dy,s,sign) in enumerate([(-.43,0,.55,1),(.51,.12,.42,-1),(0,-.79,.32,-1)]):
            cx=x+dx*r; cy=y+dy*r; radius=r*s; a=t*.65*sign/s
            pts=[]
            for k in range(80):
                theta=a+k*math.tau/80; rr=radius*(1 if k%4 in (1,2) else .84)
                pts.append((cx+math.cos(theta)*rr,cy+math.sin(theta)*rr*.75))
            shadow=[(px+8,py+14) for px,py in pts]; d.polygon(shadow,fill=(14,26,36))
            d.polygon(pts,fill=(40+15*i,80+20*i,105+22*i)); d.line(pts+[pts[0]],fill=(125,170,185),width=2)
            for k in range(5):
                theta=a+k*math.tau/5; px=cx+math.cos(theta)*radius*.48; py=cy+math.sin(theta)*radius*.36
                d.ellipse((px-radius*.12,py-radius*.1,px+radius*.12,py+radius*.1),fill=(10,22,32))
            d.ellipse((cx-12,cy-9,cx+12,cy+9),fill=self.color)
    def draw_signal_flow(self,d,x,y,r,t,g):
        chip=(x-r*.38,y-r*.32,x+r*.38,y+r*.32)
        for i in range(7):
            yy=y+(i-3)*r*.15; points=[(x-r*1.3,yy-r*.25),(x-r*.7,yy-r*.25),(x-r*.55,yy),(x-r*.38,yy)]
            d.line(points,fill=(45,90,105),width=3)
            d.line([(2*x-px,py) for px,py in points],fill=(45,90,105),width=3)
            f=(t*.45+i*.14)%1; px=x-r*1.3+f*r*2.6
            d.ellipse((px-4,yy-4,px+4,yy+4),fill=(110,225,215))
        d.rounded_rectangle((chip[0]+7,chip[1]+12,chip[2]+7,chip[3]+12),radius=16,fill=(8,18,25))
        d.rounded_rectangle(chip,radius=16,fill=(35,67,85),outline=self.color,width=3)
        for j in range(4):
            for k in range(4):
                xx=x-r*.25+k*r*.165; yy=y-r*.2+j*r*.135; light=(math.sin(t*3+j+k)>0)
                d.rounded_rectangle((xx-9,yy-7,xx+9,yy+7),radius=3,fill=(75,190,185) if light else (35,90,110))
    def draw_pathway(self,d,x,y,r,t,g):
        origin=(x-r*1.05,y+r*.55)
        for j in range(4):
            end=(x+r*.9,y+r*(-.85+j*.55)); bend=(x-r*.12,y+r*(-.6+j*.35))
            points=[origin,bend,end]; d.line(points,fill=(55,85,100),width=5)
            f=(t*.22+j*.22)%1
            a,b=(origin,bend) if f<.5 else (bend,end); q=(f*2)%1
            px=a[0]+(b[0]-a[0])*q; py=a[1]+(b[1]-a[1])*q
            d.ellipse((px-7,py-7,px+7,py+7),fill=self.color)
            size=r*.16; lift=math.sin(t*1.3+j)*5
            d.rounded_rectangle((end[0]-size,end[1]-size+lift,end[0]+size,end[1]+size+lift),radius=12,fill=(30,63,82),outline=(100,155,180),width=2)
            # Different engineering instruments, rather than invented numerical charts.
            if j==0: d.line((end[0]-size*.6,end[1]+lift,end[0],end[1]-size*.5+lift,end[0]+size*.6,end[1]+lift),fill=(120,215,215),width=3)
            elif j==1: d.ellipse((end[0]-size*.5,end[1]-size*.5+lift,end[0]+size*.5,end[1]+size*.5+lift),outline=(225,190,110),width=3)
            else: d.line((end[0]-size*.6,end[1]+size*.4+lift,end[0]+size*.6,end[1]-size*.4+lift),fill=(135,185,240),width=3)
    def draw_assembly(self,d,x,y,r,t,g):
        for i in range(5):
            spread=.18+.14*(.5+.5*math.sin(t*.9)); yy=y+(i-2)*r*spread
            pts=[(x,yy-r*.28),(x+r*.85,yy),(x,yy+r*.28),(x-r*.85,yy)]
            d.polygon([(px,py+10) for px,py in pts],fill=(12,28,40))
            d.polygon(pts,fill=(28+i*7,55+i*9,75+i*10)); d.line(pts+[pts[0]],fill=self.color,width=2)
            f=(t*.3+i*.13)%1; d.ellipse((x-r*.6+f*r*1.2-4,yy-4,x-r*.6+f*r*1.2+4,yy+4),fill=(180,230,230))
    def draw_network(self,d,x,y,r,t,g):
        pts=[]
        for i in range(16):
            theta=i*math.tau/16+t*.16; z=math.sin(theta); px=x+math.cos(theta)*r; py=y+math.sin(theta*2)*r*.55
            pts.append((px,py,z))
        for i,(px,py,z) in enumerate(pts):
            q=pts[(i+5)%16]; d.line((px,py,q[0],q[1]),fill=(40,80,100),width=2)
            f=(t*.35+i*.11)%1; sx=px+(q[0]-px)*f; sy=py+(q[1]-py)*f; d.ellipse((sx-3,sy-3,sx+3,sy+3),fill=self.color)
        for px,py,z in sorted(pts,key=lambda p:p[2]):
            rr=7+(z+1)*5; d.ellipse((px-rr,py-rr,px+rr,py+rr),fill=(80+int(z*20),155+int(z*25),180+int(z*25)))

class Typography:
    def __init__(self,brief,script,timings,size):
        self.brief=brief; self.script=script; self.timings=timings; self.size=size
        self.cues=caption_cues(script,timings); self.starts=[c['start'] for c in self.cues]; self.cache={}
        W,H=brief.size; self.vertical=H>W
        self.brand=Image.new('RGBA',(W,H)); d=ImageDraw.Draw(self.brand)
        if brief.brand=='somaiya':
            import os
            top=265 if self.vertical else 55
            for key,mw,mh,x in [('left',350,108,76),('right',125,95,W-201)]:
                logo=fit_image(Image.open(ROOT/'assets/brand'/BRAND[key]).convert('RGBA'),mw,mh)
                d.rounded_rectangle((x-10,top-8,x+logo.width+10,top+logo.height+8),radius=10,fill='white'); self.brand.alpha_composite(logo,(x,top))
            watermark=fit_image(Image.open(ROOT/'assets/brand'/BRAND['watermark']).convert('RGBA'),W*.43,H*.39)
            opacity=max(.08,min(.45,float(os.getenv('WATERMARK_OPACITY','.24')))); watermark.putalpha(round(255*opacity))
            self.brand.alpha_composite(watermark,((W-watermark.width)//2,(H-watermark.height)//2))
            textblock(d,'Somaiya Vidyavihar University',(76,1528 if self.vertical else 1000,W-152,40),28,24,align='center',fill=(190,205,215))
        self.brand=self.brand.resize(size,Image.Resampling.LANCZOS)
    def panel(self,key,text,role):
        if key in self.cache: return self.cache[key]
        W,H=self.brief.size; im=Image.new('RGBA',(W,H)); d=ImageDraw.Draw(im); box=None
        if role=='caption':
            box=(80,1350 if self.vertical else 820,W-160,150)
            d.rounded_rectangle((box[0]-16,box[1]-12,box[0]+box[2]+16,box[1]+box[3]+8),radius=18,fill=(4,12,20,190))
            info=textblock(d,text,box,62,48,align='center')
        elif role=='title':
            box=(80,440 if self.vertical else 210,W-160,150)
            d.rounded_rectangle((box[0]-16,box[1]-12,box[0]+box[2]+16,box[1]+box[3]+8),radius=18,fill=(4,12,20,155))
            info=textblock(d,text,box,58,48)
        else:
            box=(80,810 if self.vertical else 440,W-160,300 if self.vertical else 260)
            info=textblock(d,text,box,62,48,align='center')
            credit = 'Created with AI agents — LY Agentic AI Lab' if self.brief.mode == 'assignment' else 'AgentReel'
            textblock(d,credit,(80,1350 if self.vertical else 840,W-160,140),40,32,align='center')
        info['role']=role; value=(im.resize(self.size,Image.Resampling.LANCZOS),info); self.cache[key]=value; return value
    def apply(self,frame,index,t,global_t,end=False):
        im=frame.convert('RGBA'); im.alpha_composite(self.brand)
        if end:
            layer,_=self.panel('end',self.script.cta,'end'); im.alpha_composite(layer)
        else:
            if t<2.0 and index==0:
                layer,_=self.panel(('title',index),self.script.beats[index].headline,'title')
                a=ease(t/.25)*(1-ease((t-1.5)/.5)); layer=layer.copy(); layer.putalpha(layer.getchannel('A').point(lambda p:round(p*a))); im.alpha_composite(layer)
            cue_index=bisect.bisect_right(self.starts,global_t)-1
            if cue_index>=0:
                cue=self.cues[cue_index]
                if cue['start']<=global_t<cue['end']:
                    layer,_=self.panel(('caption',cue_index),cue['text'],'caption'); im.alpha_composite(layer)
        return im.convert('RGB')

def render(ctx,brief,script,board,timings,media):
    started=time.monotonic()
    size=brief.size
    theme=getattr(board,'theme','midnight')
    scenes=[Scene(brief,b,s,i,size,media_for_beat(media,i,b),timings[i]['end']-timings[i]['start'],theme) for i,(b,s) in enumerate(zip(script.beats,board.shots))]
    for scene in scenes: scene.seed=ctx.seed
    typography=Typography(brief,script,timings,size); starts=[t['start'] for t in timings]+[brief.duration_s-4]
    output=ctx.path/'assets/continuous_visuals.mp4'; err=ctx.path/'logs/motion_encoder.log'
    args=[executable('ffmpeg'),'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r',str(brief.fps),'-i','pipe:0','-an','-vf',f'scale={brief.size[0]}:{brief.size[1]}:flags=lanczos','-c:v','libx264','-preset','fast','-crf','18','-profile:v','high','-pix_fmt','yuv420p','-r',str(brief.fps),'-fps_mode','cfr','-video_track_timescale','90000','-maxrate','7M','-bufsize','14M','-threads','4',str(output)]
    review_frames={round((seg['start']+(seg['end']-seg['start'])*fraction)*brief.fps):(i,label) for i,seg in enumerate(timings) for fraction,label in ((.15,'early'),(.55,'middle'),(.9,'late'))}
    frames=brief.duration_s*brief.fps; proc=None
    try:
        with err.open('wb') as log:
            proc=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=log)
            for number in range(frames):
                if number%brief.fps==0: ctx.check()
                g=number/brief.fps; index=bisect.bisect_right(starts,g)-1; end=index==len(scenes)
                index=min(index,len(scenes)-1); local=g-starts[index]
                scene=scenes[index]; frame=scene.frame(local,g)
                if not end and index and local<.4 and scene.shot.transition=='dissolve':
                    previous=scenes[index-1].frame(g-starts[index-1],g)
                    frame=Image.blend(previous,frame,ease(local/.4))
                if end: frame=ImageEnhance.Brightness(frame).enhance(.55)
                frame=typography.apply(frame,index,local,g,end)
                if scene.footage and scene.footage.item.get('kind') not in ('official','self_shot','generated'):
                    d=ImageDraw.Draw(frame); d.text((54,400 if size[1]>size[0] else 160),'ILLUSTRATIVE FOOTAGE',font=font(28),fill=(225,235,240))
                if scene.illustration:
                    label=scene.illustration.label(local)
                    d=ImageDraw.Draw(frame); d.text((54,400 if size[1]>size[0] else 160),label,font=font(28),fill=(225,235,240))
                if number in (brief.fps*2,brief.fps*12,brief.fps*(brief.duration_s-2)):
                    frame.resize(brief.size,Image.Resampling.LANCZOS).save(ctx.path/'assets'/f'motion_preview_{number//brief.fps:02}.jpg',quality=94)
                if number in review_frames:
                    shot_index,label=review_frames[number]
                    frame.resize((540,960) if size[1]>size[0] else (960,540),Image.Resampling.LANCZOS).save(ctx.path/'assets'/f'scene_{shot_index:02}_{label}.jpg',quality=90)
                proc.stdin.write(frame.tobytes())
            proc.stdin.close(); proc.wait(timeout=120)
            if proc.returncode: raise RuntimeError('Motion encoder failed: '+err.read_text(errors='replace')[-1500:])
    finally:
        if proc and proc.poll() is None: proc.kill(); proc.wait()
        for scene in scenes:
            if scene.footage: scene.footage.close()
            if scene.illustration: scene.illustration.close()
    layout=[]
    for i in range(len(scenes)):
        boxes=[typography.panel(('title',i),script.beats[i].headline,'title')[1]] if i==0 else []
        boxes.extend(typography.panel(('caption',j),c['text'],'caption')[1] for j,c in enumerate(typography.cues) if c['beat_index']==i)
        layout.append({'shot':i,'text':boxes,'branding':brief.brand})
    info=typography.panel('end',script.cta,'end')[1]
    layout.append({'shot':'end_card','text':[info,{'role':'disclosure','box':[80,1350 if typography.vertical else 840,brief.size[0]-160,140],'font_size':40}],'branding':brief.brand})
    ctx.artifact('layout.json',layout,'asset_generator')
    renderer = 'rainbow_optics' if all(s.kind == 'rainbow_optics' for s in scenes) else ('scene_graph_v1' if all(s.shot.scene is not None for s in scenes) else 'continuous_scene_v2')
    ctx.artifact('motion_plan.json',{'renderer':renderer,'internal_resolution':list(size),'output_resolution':list(brief.size),'transition_s':.4,'caption_alignment':'phrase estimates within measured beat audio; not forced word alignment','scenes':[{'beat_index':i,'demonstration':s.kind,'camera':s.shot.camera,'transition':s.shot.transition,'visible_change':s.shot.scene.visible_change if s.shot.scene else None,'objects':[o.subject for o in s.shot.scene.objects] if s.shot.scene else [],'media':str(s.footage.path) if s.footage else ('ai_sequence:'+'+'.join(s.illustration.kinds) if s.illustration else None),'illustrative':True if not s.footage else s.footage.item.get('kind') not in ('official','self_shot')} for i,s in enumerate(scenes)]},'editor')
    manifest=ctx.path/'assets/manifest.csv'
    fieldnames=['file','source','licence','kind','sha256','beat_index','description','start_s','identifiable_people','consent']
    with manifest.open('a',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fieldnames)
        if stream.tell()==0: writer.writeheader()
        for path in [output,*sorted((ctx.path/'assets').glob('motion_preview_*.jpg'))]:
            writer.writerow({'file':str(path.relative_to(ctx.path)),'source':'AgentReel scene renderer; see motion_plan.json','licence':'Original procedural visuals; supplied media retain the licences in this manifest','kind':'generated','sha256':sha(path)})
    ctx.emit('editor','render_continuous_scenes',latency_s=round(time.monotonic()-started,3),inputs={'frames':frames,'renderer':renderer},output_path=str(output.relative_to(ctx.path)),output_sha256=sha(output))
    return output
