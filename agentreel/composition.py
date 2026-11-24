"""Topic-independent scene graph: timed geometry, depth, lighting and directed flows.

Plans are validated data, never executable model-generated Python. Coordinates live
inside a caption-safe stage. There are no topic names or prompt-specific branches.
"""
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageColor, ImageFilter, ImageFont
from .visuals import font
from . import icons

# Formula glyphs missing from Noto Sans come from a system symbol font; without one,
# readable ASCII stands in rather than an empty box.
SYMBOL_FONTS=[os.getenv('SYMBOL_FONT_PATH',''),'C:/Windows/Fonts/seguisym.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/System/Library/Fonts/Apple Symbols.ttf']
ASCII={'→':'->','←':'<-','⇌':'<=>','↔':'<->','√':'sqrt','⟶':'->'}
_missing={}

def missing(ft,ch):
    key=(ft.path,ch)
    if key not in _missing: _missing[key]=bytes(ft.getmask(ch))==bytes(ft.getmask('󰀀'))
    return _missing[key]

def runs(text,ft):
    """Split text into (segment, font) runs, swapping in a symbol font where needed."""
    symbol=next((p for p in SYMBOL_FONTS if p and os.path.isfile(p)),None)
    result=[]
    for ch in text:
        use=ft
        if not ch.isspace() and missing(ft,ch):
            alt=ImageFont.truetype(symbol,ft.size) if symbol else None
            if alt and not missing(alt,ch): use=alt
            else: ch=ASCII.get(ch,'?')
        if result and result[-1][1].path==use.path: result[-1]=(result[-1][0]+ch,result[-1][1])
        else: result.append((ch,use))
    return result

def rich_length(draw,text,ft):
    return sum(draw.textlength(s,font=f) for s,f in runs(text,ft))

def rich_text(draw,xy,text,ft,fill):
    x,y=xy
    for s,f in runs(text,ft):
        draw.text((x,y),s,font=f,fill=fill); x+=draw.textlength(s,font=f)


def pose_at(obj, progress):
    p=max(0.,min(1.,progress))
    a,b=obj.poses[0],obj.poses[-1]
    for left,right in zip(obj.poses,obj.poses[1:]):
        if left.time<=p<=right.time:
            a,b=left,right; break
    u=(p-a.time)/max(.0001,b.time-a.time)
    # Smooth ends prevent teleporting and mechanical slide-in transitions.
    u=u*u*(3-2*u)
    return {name:getattr(a,name)+(getattr(b,name)-getattr(a,name))*u
            for name in ('x','y','scale','angle','opacity')}


def ease(x):
    x=max(0.,min(1.,x)); return x*x*(3-2*x)


# One colour world per film. Light themes cast shadows instead of glowing.
THEMES={
    'midnight':dict(top=(8,14,28),bottom=(24,34,56),glow=(70,110,170),line=(52,72,102),mote=(170,200,225),light=False),
    'ocean':dict(top=(3,22,40),bottom=(6,62,86),glow=(50,170,200),line=(30,96,124),mote=(170,230,240),light=False),
    'sunset':dict(top=(34,14,42),bottom=(120,50,56),glow=(255,150,90),line=(130,64,78),mote=(255,210,170),light=False),
    'forest':dict(top=(6,24,18),bottom=(24,60,38),glow=(130,210,120),line=(44,90,62),mote=(200,240,180),light=False),
    'neon':dict(top=(8,4,22),bottom=(30,8,52),glow=(210,70,255),line=(84,40,150),mote=(120,240,255),light=False),
    'paper':dict(top=(244,239,228),bottom=(222,212,196),glow=(255,255,250),line=(204,192,174),mote=(176,160,136),light=True),
    'ember':dict(top=(18,8,6),bottom=(66,22,10),glow=(255,120,40),line=(100,46,28),mote=(255,180,110),light=False),
    'arctic':dict(top=(226,238,246),bottom=(168,198,220),glow=(255,255,255),line=(150,180,205),mote=(255,255,255),light=True),
}
# Faint background props establish where a scene happens without competing with it.
PROPS={'lab':('flask','atom','flask'),'digital':('chip','screen','chip'),'sky':('cloud','cloud','sun'),
       'city':('building','building','building'),'ocean':('drop','drop','drop'),'micro':('cell','cell','virus')}
# Ambient particles per setting: (count, x speed, y speed, size, style).
AMBIENT={'ocean':(26,4,-46,5,'bubble'),'digital':(30,0,0,4,'pulse'),'micro':(22,10,-6,16,'soft'),
         'sky':(14,22,0,3,'mote'),'space':(40,6,-2,2,'mote')}


def lerp(a,b,t): return tuple(round(x+(y-x)*t) for x,y in zip(a,b))


class Composition:
    # Objects read clearly on a phone; the stage stays above the caption band.
    OBJECT_SCALE=1.7

    def __init__(self, plan, size, camera, duration, theme='midnight', overlay=False):
        # overlay=True draws only text plates and ambient motes on a transparent layer,
        # for compositing over footage-like stills.
        self.overlay=overlay
        self.plan=plan; self.size=size; self.camera=camera; self.duration=max(.1,duration)
        self.theme=THEMES.get(theme,THEMES['midnight']); th=self.theme
        self.sprites={}; self.glows={}; self.fonts={}
        w,h=size; self.vertical=h>w
        self.stage=(w*.08,h*(.27 if self.vertical else .18),w*.84,h*(.40 if self.vertical else .52))
        self.background=Image.new('RGBA',size) if overlay else self.backdrop(plan.setting)
        self.offsets=self.separate()
        rng=np.random.default_rng(17)
        count,self.vx,self.vy,self.mote_size,self.mote_style=AMBIENT.get(plan.setting,(36,14,-9,3,'mote'))
        # Warm themes send their motes upward like sparks.
        if theme in ('ember','sunset'): self.vy-=18
        self.motes=np.stack([rng.uniform(0,w,count),rng.uniform(0,h,count),rng.uniform(.25,1,count)],axis=1)
        for obj in plan.objects:
            if obj.shape=='text': continue
            sprite=icons.draw(obj.shape,obj.color); self.sprites[obj.id]=sprite
            alpha=Image.new('L',(576,576)); alpha.paste(sprite.getchannel('A'),(96,96))
            alpha=alpha.filter(ImageFilter.GaussianBlur(30))
            tint=(0,0,0) if th['light'] else ImageColor.getrgb(obj.color)
            halo=Image.new('RGBA',alpha.size,(*tint,0)); halo.putalpha(alpha.point(lambda v:round(v*(.32 if th['light'] else .55))))
            self.glows[obj.id]=halo
        self.phase={obj.id:(sum(map(ord,obj.id))%97)/97*math.tau for obj in plan.objects}

    def backdrop(self,setting):
        th=self.theme; w,h=self.size
        yy,xx=np.mgrid[0:h,0:w]
        vertical=(yy/h)[...,None]
        base=np.array(th['top'],float)*(1-vertical)+np.array(th['bottom'],float)*vertical
        glow=np.exp(-(((xx/w-.5)/.6)**2+((yy/h-.42)/.45)**2))[...,None]
        base=base+(np.array(th['glow'],float)-base)*glow*(.12 if th['light'] else .22)
        vignette=(1-(.18 if th['light'] else .42)*np.clip(((xx/w-.5)**2+(yy/h-.5)**2)*2.2,0,1))[...,None]
        im=Image.fromarray(np.clip(base*vignette,0,255).astype('uint8')).convert('RGBA')
        d=ImageDraw.Draw(im); line=(*th['line'],255)
        floor=int(self.stage[1]+self.stage[3]*1.12); rng=np.random.default_rng(23)
        if setting in ('studio','interior','lab'):
            # Faint perspective grid gives depth without competing with the subject.
            for j in range(-10,11): d.line((w*.5+j*w*.04,floor,w*.5+j*w*.22,h),fill=line,width=1)
            for k in range(8):
                y=floor+(h-floor)*(k/8)**1.7; d.line((0,y,w,y),fill=line,width=1)
        if setting=='digital':
            step=round(w/14)
            for x in range(0,w,step): d.line((x,0,x,h),fill=line,width=1)
            for y in range(0,h,step): d.line((0,y,w,y),fill=line,width=1)
        if setting=='landscape':
            d.polygon([(0,floor),(w*.3,floor-h*.03),(w*.62,floor+h*.01),(w,floor-h*.025),(w,h),(0,h)],fill=(*lerp(th['bottom'],(0,0,0),.25),255))
        if setting=='ocean':
            for k in range(6):
                y=h*.08+k*h*.035; d.line((0,y,w,y+h*.01),fill=(*lerp(th['bottom'],(255,255,255),.12),255),width=3)
            d.polygon([(0,floor),(w*.4,floor-h*.02),(w,floor+h*.01),(w,h),(0,h)],fill=(*lerp(th['top'],(0,0,0),.2),255))
        if setting=='city':
            x=0
            while x<w:
                bw=rng.uniform(.07,.14)*w; bh=rng.uniform(.06,.2)*h; tone=(*lerp(th['bottom'],(0,0,0),.35),255)
                d.rectangle((x,floor-bh,x+bw,h),fill=tone)
                for wy in np.arange(floor-bh+12,floor-8,26):
                    for wx in np.arange(x+8,x+bw-10,22):
                        if rng.random()<.35: d.rectangle((wx,wy,wx+8,wy+10),fill=(*lerp(th['glow'],(255,255,255),.3),255))
                x+=bw+2
        if setting=='space':
            for x,y,r in zip(rng.uniform(0,w,150),rng.uniform(0,h,150),rng.uniform(.6,2.2,150)):
                d.ellipse((x-r,y-r,x+r,y+r),fill=(150,165,190))
        props=PROPS.get(setting)
        if props:
            layer=Image.new('RGBA',(w,h))
            spots=[(.14,.2,.22),(.86,.16,.18),(.82,.86,.24)] if self.vertical else [(.08,.25,.3),(.92,.2,.26),(.9,.82,.3)]
            for shape,(px,py,s) in zip(props,spots):
                sp=icons.draw(shape,'#%02x%02x%02x'%lerp(th['line'],th['glow'],.4)).resize((round(w*s),)*2,Image.Resampling.LANCZOS)
                sp.putalpha(sp.getchannel('A').point(lambda v:round(v*.22)))
                layer.alpha_composite(sp,(round(px*w-sp.width/2),round(py*h-sp.height/2)))
            im.alpha_composite(layer.filter(ImageFilter.GaussianBlur(3)))
        if setting=='micro':
            layer=Image.new('RGBA',(w,h)); g=ImageDraw.Draw(layer)
            for x,y,r in zip(rng.uniform(0,w,14),rng.uniform(0,h,14),rng.uniform(.05,.14,14)*w):
                g.ellipse((x-r,y-r,x+r,y+r),fill=(*th['glow'],34),outline=(*th['glow'],70),width=4)
            im.alpha_composite(layer.filter(ImageFilter.GaussianBlur(10)))
        return im

    def separate(self):
        """Horizontal nudges for objects that START on top of each other. Converging later
        in the shot is a deliberate action (entering, combining) and is left alone."""
        sx,sy,sw,sh=self.stage; unit=min(sw,sh)*self.OBJECT_SCALE*.9
        solid=[o for o in self.plan.objects if o.shape!='text' and o.poses[0].opacity>.05]
        offsets={o.id:0. for o in self.plan.objects}
        for _ in range(3):
            for i,a in enumerate(solid):
                for b in solid[i+1:]:
                    pa,pb=a.poses[0],b.poses[0]
                    # Containers may hold their contents; overlap is the point there.
                    if a.shape in ('disc','cell','vessel') or b.shape in ('disc','cell','vessel'): continue
                    dx=(pb.x+offsets[b.id]-pa.x-offsets[a.id])*sw; dy=(pb.y-pa.y)*sh
                    need=(a.size*pa.scale+b.size*pb.scale)*unit
                    gap=need-math.hypot(dx,dy)
                    if gap<=0: continue
                    push=gap/sw/2*(1 if dx>=0 else -1)
                    offsets[a.id]-=push; offsets[b.id]+=push
        # Keep nudged objects on stage.
        for o in self.plan.objects:
            lo,hi=min(p.x for p in o.poses),max(p.x for p in o.poses)
            offsets[o.id]=max(.08-lo,min(.92-hi,offsets[o.id]))
        return offsets

    def positions(self, progress):
        sx,sy,sw,sh=self.stage; result={}
        for obj in self.plan.objects:
            pose=pose_at(obj,progress); pose['x']+=self.offsets.get(obj.id,0); depth=.75+obj.depth*.4
            zoom=1+(.09*progress if self.camera=='push_in' else .025*progress)
            drift=(-1 if self.camera=='track_left' else 1 if self.camera=='track_right' else 0)*progress*sw*.04*depth
            if self.camera=='orbit': drift=math.sin(progress*math.pi)*sw*.035*(depth-.95)*4
            x=sx+sw*.5+(pose['x']-.5)*sw*zoom+drift
            y=sy+sh*.5+(pose['y']-.5)*sh*zoom
            result[obj.id]=(obj,pose,x,y,obj.size*min(sw,sh)*pose['scale']*depth*zoom*self.OBJECT_SCALE)
        return result

    def text_font(self,label,limit,base):
        """Largest size that keeps a formula on one line inside the stage."""
        key=(label,round(base))
        if key not in self.fonts:
            probe=ImageDraw.Draw(Image.new('RGBA',(8,8))); size=int(base)
            while size>30 and rich_length(probe,label,font(size,label))>limit: size-=4
            self.fonts[key]=font(size,label)
        return self.fonts[key]

    def ambient(self,g,t):
        w,h=self.size; tone=self.theme['mote']
        for x,y,z in self.motes:
            px=(x+t*self.vx*z)%w; py=(y+t*self.vy*z)%h; r=self.mote_size*(.5+z)
            if self.mote_style=='bubble':
                g.ellipse((px-r,py-r,px+r,py+r),outline=(*tone,round(70+90*z)),width=2)
            elif self.mote_style=='pulse':
                # Data pulses run along the backdrop grid.
                step=w/14; lane=round(y/step)*step; q=(x/w+t*.12*z)%1
                g.rectangle((q*w-r*3,lane-r*.5,q*w+r*3,lane+r*.5),fill=(*tone,round(90+120*z)))
            elif self.mote_style=='soft':
                g.ellipse((px-r,py-r,px+r,py+r),fill=(*tone,round(20+30*z)))
            else:
                g.ellipse((px-r,py-r,px+r,py+r),fill=(*tone,round(40+70*z)))

    def frame(self,t):
        progress=max(0,min(1,t/self.duration)); th=self.theme
        im=self.background.copy(); fx=Image.new('RGBA',self.size); ui=Image.new('RGBA',self.size)
        d=ImageDraw.Draw(im); g=ImageDraw.Draw(fx); u=ImageDraw.Draw(ui)
        self.ambient(g,t)
        plate=(250,250,246) if th['light'] else (6,14,24); ink=(24,30,40) if th['light'] else (245,248,252)
        positions=self.positions(progress); placed=[]
        # Directed connections are tied to live object positions, not floating decoration.
        for link in ([] if self.overlay else self.plan.links):
            if not link.start<=progress<=link.end: continue
            a,b=positions[link.source],positions[link.target]
            dx,dy=b[2]-a[2],b[3]-a[3]; length=math.hypot(dx,dy)
            if length<1: continue
            ux,uy=dx/length,dy/length
            x1,y1=a[2]+ux*a[4]*.8,a[3]+uy*a[4]*.8;x2,y2=b[2]-ux*b[4]*.8,b[3]-uy*b[4]*.8
            color=ImageColor.getrgb(link.color)
            if link.style=='ray':
                # A beam that travels from source to target, then keeps a bright pulse moving.
                local=(progress-link.start)/max(.001,link.end-link.start)
                reach=ease(min(1,local*2.2)); ex,ey=x1+(x2-x1)*reach,y1+(y2-y1)*reach
                for width,alpha in ((34,40),(18,90),(7,255)):
                    g.line((x1,y1,ex,ey),fill=(*color,alpha),width=width)
                if reach>=1:
                    q=(t*.7)%1; px,py=x1+(x2-x1)*q,y1+(y2-y1)*q
                    g.ellipse((px-11,py-11,px+11,py+11),fill=(255,255,240,230))
                continue
            g.line((x1,y1,x2,y2),fill=(*color,70),width=12)
            d.line((x1,y1,x2,y2),fill=tuple(round(c*.7) for c in color),width=4)
            if link.style in ('arrow','flow'):
                r=20;d.polygon([(x2,y2),(x2-ux*r-uy*r*.55,y2-uy*r+ux*r*.55),(x2-ux*r+uy*r*.55,y2-uy*r-ux*r*.55)],fill=color)
            if link.style=='flow':
                for j in range(5):
                    q=(t*.3+j/5)%1;x=x1+(x2-x1)*q;y=y1+(y2-y1)*q
                    g.ellipse((x-9,y-9,x+9,y+9),fill=(*color,255))
        for obj,pose,x,y,r in sorted(positions.values(),key=lambda row:row[0].depth):
            if pose['opacity']<=.01: continue
            a=round(255*pose['opacity'])
            if obj.shape=='text':
                limit=self.stage[2]*1.02
                ft=self.text_font(obj.label,limit,(84 if self.vertical else 72)*max(.6,min(1.4,pose['scale'])))
                tw=rich_length(u,obj.label,ft); fs=ft.size
                accent=ImageColor.getrgb(obj.color)
                u.rounded_rectangle((x-tw/2-26,y-fs*.75,x+tw/2+26,y+fs*.85),radius=18,fill=(*plate,round(a*.86)),outline=(*accent,a),width=4)
                rich_text(u,(x-tw/2,y-fs*.62),obj.label,ft,(*ink,a))
                placed.append((x-tw/2-26,y-fs*.75,x+tw/2+26,y+fs*.85))
                continue
            if self.overlay: continue
            phase=self.phase[obj.id]
            # Idle life: a gentle float, plus spin or flicker where the shape suggests it.
            y+=math.sin(t*1.4+phase)*r*.04
            if obj.shape=='flame': r*=1+.05*math.sin(t*9+phase)
            angle=pose['angle']+icons.SPIN.get(obj.shape,0)*t
            n=max(4,min(1600,round(r*2)))
            sprite=self.sprites[obj.id].resize((n,n),Image.Resampling.BICUBIC)
            if obj.shape in icons.FLIP:
                sprite=sprite.resize((max(4,round(n*(.25+.75*abs(math.cos(t*1.2+phase))))),n),Image.Resampling.BICUBIC)
            if angle: sprite=sprite.rotate(angle,resample=Image.Resampling.BICUBIC,expand=True)
            halo=self.glows[obj.id].resize((round(n*1.5),round(n*1.5)),Image.Resampling.BILINEAR)
            if pose['opacity']<.99:
                sprite.putalpha(sprite.getchannel('A').point(lambda v:round(v*pose['opacity'])))
                halo.putalpha(halo.getchannel('A').point(lambda v:round(v*pose['opacity'])))
            # Groups: copies on a loose grid around the pose, each bobbing on its own beat.
            count=getattr(obj,'count',1); cols=math.ceil(math.sqrt(count)); rows=math.ceil(count/cols); step=0
            if count>1:
                sprite=sprite.resize((max(4,round(sprite.width*.62)),max(4,round(sprite.height*.62))),Image.Resampling.BICUBIC)
                halo=halo.resize((max(4,round(halo.width*.62)),max(4,round(halo.height*.62))),Image.Resampling.BILINEAR)
                step=n*.62*1.08
            for k in range(count):
                cx=x+((k%cols)-(cols-1)/2)*step+((k//cols)%2)*step*.3
                cy=y+((k//cols)-(rows-1)/2)*step*.9+(math.sin(t*1.7+k*1.3)*r*.03 if count>1 else 0)
                hx,hy=(cx+r*.08,cy+r*.14) if th['light'] else (cx,cy)
                im.alpha_composite(halo,(round(hx-halo.width/2),round(hy-halo.height/2)))
                im.alpha_composite(sprite,(round(cx-sprite.width/2),round(cy-sprite.height/2)))
            if obj.label:
                fs=44 if self.vertical else 36
                key=(fs,obj.label)
                if key not in self.fonts:self.fonts[key]=font(fs,obj.label)
                ft=self.fonts[key];tw=rich_length(u,obj.label,ft)
                # Object labels are subordinate to narration, never full-screen cards.
                yy=y+(r if count==1 else (rows-1)*step*.45+n*.31)+10
                # Step below any label already placed this frame instead of overprinting it.
                for _ in range(4):
                    box=(x-tw/2-12,yy-4,x+tw/2+12,yy+fs*1.38)
                    hit=next((b for b in placed if box[0]<b[2] and b[0]<box[2] and box[1]<b[3] and b[1]<box[3]),None)
                    if not hit: break
                    yy=hit[3]+8
                placed.append((x-tw/2-12,yy-4,x+tw/2+12,yy+fs*1.38))
                u.rounded_rectangle((x-tw/2-12,yy-4,x+tw/2+12,yy+fs*1.38),radius=10,fill=(*plate,round(a*.88)))
                rich_text(u,(x-tw/2,yy),obj.label,ft,(*ink,a))
        im.alpha_composite(fx); im.alpha_composite(ui)
        return im if self.overlay else im.convert('RGB')
