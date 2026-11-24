import math,os,random,re
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .runtime import ROOT,dump

ACCENTS={'red':(184,43,55),'teal':(38,153,149),'gold':(206,158,66),'blue':(71,126,206)}
BRAND={'left':'10000000000001920000007B9293135B.jpg','right':'10000001000000E9000000AEE52F4A92.png','watermark':'100000000000012C000001BBA674B6FF.jpg'}
def font(size,text=''):
    devanagari=bool(re.search(r'[\u0900-\u097f]',text))
    name='NotoSansDevanagari.ttf' if devanagari else 'NotoSans.ttf'
    configured=os.getenv('DEVANAGARI_FONT_PATH' if devanagari else 'FONT_PATH','')
    path=Path(configured) if configured else ROOT/'assets/fonts'/name
    if not path.exists(): raise FileNotFoundError('Missing font '+str(path)+'. Run scripts/setup_models.py.')
    return ImageFont.truetype(str(path),int(size))
def wrap(draw,text,ft,width):
    lines=[]; current=''
    for word in text.split():
        trial=(current+' '+word).strip()
        if draw.textlength(trial,font=ft)>width and current: lines.append(current); current=word
        else: current=trial
    if current: lines.append(current)
    return lines
def textblock(draw,text,box,size=60,min_size=48,fill='white',align='left'):
    x,y,w,h=box
    for s in range(size,min_size-1,-2):
        ft=font(s,text); lines=wrap(draw,text,ft,w); leading=int(s*1.35)
        if len(lines)*leading<=h and all(draw.textlength(line,font=ft)<=w for line in lines): break
    else: raise ValueError('Text does not fit safe area: '+text[:80])
    for i,line in enumerate(lines):
        dx=x+(w-draw.textlength(line,font=ft))/2 if align=='center' else x
        draw.text((dx,y+i*leading),line,font=ft,fill=fill,stroke_width=0)
    return {'text':text,'box':list(box),'font_size':s,'line_count':len(lines)}
def fit_image(image,w,h):
    image=image.copy(); image.thumbnail((int(w),int(h)),Image.Resampling.LANCZOS); return image

def background(size,accent,seed,visual):
    w,h=size; color=np.array(ACCENTS[accent],dtype=float); yy,xx=np.mgrid[0:h,0:w]
    glow=np.exp(-(((xx-w*.72)/(w*.65))**2+((yy-h*.5)/(h*.45))**2))
    base=np.zeros((h,w,3),dtype=np.uint8)
    for c in range(3): base[:,:,c]=np.clip(12+glow*color[c]*.26+yy/h*5,0,255).astype(np.uint8)
    im=Image.fromarray(base).convert('RGB'); d=ImageDraw.Draw(im)
    for x in range(0,w,90): d.line((x,0,x,h),fill=(25,31,39),width=1)
    for y in range(0,h,90): d.line((0,y,w,y),fill=(25,31,39),width=1)
    rng=random.Random(seed)
    for _ in range(35):
        x=rng.randrange(w); y=rng.randrange(h); d.ellipse((x,y,x+3,y+3),fill=(90,100,110))
    return im

def diagram(draw,size,kind,color):
    w,h=size; cx=w*.5; cy=h*.53; r=min(w,h)*.24
    muted=tuple(int(v*.5) for v in color)
    if kind=='orbit':
        for scale in (.45,.7,1): draw.ellipse((cx-r*scale,cy-r*scale,cx+r*scale,cy+r*scale),outline=muted,width=3)
        for angle in (30,150,270):
            x=cx+r*math.cos(math.radians(angle)); y=cy+r*math.sin(math.radians(angle))
            draw.ellipse((x-13,y-13,x+13,y+13),fill=color)
        draw.ellipse((cx-38,cy-38,cx+38,cy+38),fill=color)
    elif kind in ('network','grid'):
        pts=[(cx+math.cos(i*math.pi/3)*r,cy+math.sin(i*math.pi/3)*r) for i in range(6)]
        for i,p in enumerate(pts):
            draw.line((cx,cy,*p),fill=muted,width=3); draw.line((*p,*pts[(i+1)%6]),fill=muted,width=3)
            draw.rounded_rectangle((p[0]-28,p[1]-28,p[0]+28,p[1]+28),radius=10,fill=color)
        draw.ellipse((cx-55,cy-55,cx+55,cy+55),outline=color,width=5)
    elif kind=='path':
        pts=[(cx-r,cy+r*.45),(cx-r*.3,cy-r*.35),(cx+r*.3,cy+r*.2),(cx+r,cy-r*.45)]
        draw.line(pts,fill=muted,width=7)
        for x,y in pts: draw.ellipse((x-20,y-20,x+20,y+20),fill=color)
    elif kind=='layers':
        for i in range(4):
            y=cy-r*.65+i*r*.35
            draw.rounded_rectangle((cx-r+i*15,y,cx+r-i*15,y+r*.24),radius=18,outline=color,width=3)
    else:
        for i in range(21):
            x=cx-r+i*r/10; amp=(.25+.75*abs(math.sin(i*.38)))*r*.7
            draw.rounded_rectangle((x-5,cy-amp,x+5,cy+amp),radius=5,fill=color)

def overlay(brief,beat,shot,index,total,end_card=False):
    w,h=brief.size; vertical=h>w; im=Image.new('RGBA',(w,h),(0,0,0,0)); d=ImageDraw.Draw(im); boxes=[]
    margin=76 if vertical else 110; top=265 if vertical else 65
    if brief.brand=='somaiya':
        for key,maxw,maxh,x in [('left',410 if vertical else 470,126 if vertical else 120,margin),('right',155,115,w-margin-155)]:
            logo=fit_image(Image.open(ROOT/'assets/brand'/BRAND[key]).convert('RGBA'),maxw,maxh)
            d.rounded_rectangle((x-14,top-10,x+logo.width+14,top+logo.height+10),radius=12,fill=(255,255,255,255))
            im.alpha_composite(logo,(int(x),int(top)))
        watermark=fit_image(Image.open(ROOT/'assets/brand'/BRAND['watermark']).convert('RGBA'),w*.43,h*.39)
        alpha=max(.08,min(.45,float(os.getenv('WATERMARK_OPACITY','.24'))))
        watermark.putalpha(watermark.getchannel('A').point(lambda p:round(p*alpha)))
        im.alpha_composite(watermark,(int((w-watermark.width)/2),int((h-watermark.height)/2)))
    d=ImageDraw.Draw(im)
    title_y=450 if vertical else 225
    label='SOMAIYA / '+shot.label.upper() if brief.brand=='somaiya' else shot.label.upper()
    boxes.append(textblock(d,label,(margin,title_y-40,w-2*margin,45),size=32,min_size=26,fill=(188,201,214)))
    title=script_end_title(brief) if end_card else beat.headline
    boxes.append(textblock(d,title,(margin,title_y,w-2*margin,230 if vertical else 150),size=82 if vertical else 76,min_size=54))
    color=ACCENTS[shot.accent]
    d.rounded_rectangle((margin,title_y+250 if vertical else title_y+150,margin+110,title_y+258 if vertical else title_y+158),radius=4,fill=color)
    if not end_card:
        capbox=(margin,1250 if vertical else 810,w-2*margin,260 if vertical else 150)
        d.rounded_rectangle((capbox[0]-20,capbox[1]-15,capbox[0]+capbox[2]+20,capbox[1]+capbox[3]+10),radius=24,fill=(6,12,20,225))
        boxes.append(textblock(d,beat.caption,capbox,size=54,min_size=48,align='center'))
        y=1218 if vertical else 770
        for j in range(total):
            length=(w-2*margin-(total-1)*10)/total; x=margin+j*(length+10)
            d.rounded_rectangle((x,y,x+length,y+5),radius=2,fill=color if j<=index else (55,63,74))
    else:
        boxes.append(textblock(d,beat.caption,(margin,850 if vertical else 500,w-2*margin,260 if vertical else 230),size=62,min_size=48,align='center'))
        credit = 'Created with AI agents — LY Agentic AI Lab' if brief.mode == 'assignment' else 'AgentReel'
        boxes.append(textblock(d,credit,(margin,1350 if vertical else 840,w-2*margin,140),size=40,min_size=32,fill=(200,208,215),align='center'))
    if brief.brand=='somaiya':
        boxes.append(textblock(d,'Somaiya Vidyavihar University',(margin,1528 if vertical else 1000,w-2*margin,40),size=28,min_size=24,fill=(170,181,194),align='center'))
    return im,boxes
def script_end_title(brief): return 'Your next step starts here.' if brief.language.lower().startswith('english') else brief.topic[:60]

def make_frames(ctx,brief,script,board):
    folder=ctx.path/'assets'; folder.mkdir(exist_ok=True); layout=[]; frames=[]
    for i,(beat,shot) in enumerate(zip(script.beats,board.shots)):
        bg=background(brief.size,shot.accent,ctx.seed+i,shot.visual)
        diagram(ImageDraw.Draw(bg),brief.size,shot.visual,ACCENTS[shot.accent])
        ov,boxes=overlay(brief,beat,shot,i,len(script.beats))
        b=folder/f'background_{i:02}.png'; o=folder/f'overlay_{i:02}.png'; bg.save(b); ov.save(o)
        preview=bg.convert('RGBA'); preview.alpha_composite(ov); preview.convert('RGB').save(folder/f'preview_{i:02}.jpg',quality=90)
        frames.append((b,o)); layout.append({'shot':i,'text':boxes,'branding':brief.brand})
    from .models import Beat,Shot
    endbeat=Beat(narration='',caption=script.cta,headline='',fact_ids=[])
    endshot=Shot(beat_index=0,visual='orbit',visual_description='End card',label='TAKE THE NEXT STEP',accent='red' if brief.brand=='somaiya' else 'teal')
    bg=background(brief.size,endshot.accent,ctx.seed+99,'orbit'); ov,boxes=overlay(brief,endbeat,endshot,0,1,True)
    b=folder/'end_background.png'; o=folder/'end_overlay.png'; bg.save(b); ov.save(o)
    preview=bg.convert('RGBA'); preview.alpha_composite(ov); preview.convert('RGB').save(folder/'end_preview.jpg',quality=90)
    frames.append((b,o)); layout.append({'shot':'end_card','text':boxes,'branding':brief.brand})
    ctx.artifact('layout.json',layout,'asset_generator'); return frames
