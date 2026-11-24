"""A round-drop cross section for a primary rainbow, using Snell's law."""
import math

from PIL import Image, ImageDraw
from .visuals import font


SPECTRUM = [("#ff877d", 1.331), ("#ffc671", 1.333), ("#8ce4b1", 1.335),
            ("#73d4ff", 1.338), ("#bfabff", 1.343)]


def sky_geometry(size):
    w,h=size
    eye=(w*.48,h*.45+w*.27)
    drops=[]
    for _,index in SPECTRUM:
        a,b=path_for(index)[-2:]
        dx,dy=b[0]-a[0],b[1]-a[1]
        drop_x=w*.80
        drops.append((drop_x, eye[1]+(drop_x-eye[0])*dy/dx))
    sun=(w*.16,h*.45)
    arcs=[(w*.5,h*.59,min(w*.37,h*.23)-i*w*.008) for i in range(5)]
    return {"eye":eye,"sun":sun,"drops":drops,"arcs_center_x_y_radius":arcs}


def path_for(index):
    # Unit circle; y increases down the screen. One internal reflection.
    incidence = math.asin(0.86)
    refracted = math.asin(0.86 / index)
    heading = incidence - refracted
    entry = (-math.cos(incidence), -0.86)
    direction = (math.cos(heading), math.sin(heading))
    chord = 2 * math.cos(refracted)
    back = tuple(a + chord * b for a, b in zip(entry, direction))
    dot = sum(a * b for a, b in zip(direction, back))
    reflected = tuple(a - 2 * dot * b for a, b in zip(direction, back))
    exit_point = tuple(a + chord * b for a, b in zip(back, reflected))
    dot = sum(a * b for a, b in zip(reflected, exit_point))
    tangent = tuple(index * (a - dot * b) for a, b in zip(reflected, exit_point))
    outgoing = tuple(a + math.cos(incidence) * b for a, b in zip(tangent, exit_point))
    end = tuple(a + 1.05 * b for a, b in zip(exit_point, outgoing))
    return [(-1.95, -0.86), entry, back, exit_point, end]


class RainbowDiagram:
    @staticmethod
    def phase(narration):
        text = narration.lower()
        if any(word in text for word in ("many drop", "behind you", "behind the", "your eyes", "your back")):
            return "observer"
        if any(word in text for word in ("curved bow", "in the sky", "rainbow arc")):
            return "arc"
        if any(word in text for word in ("exit", "way out", "leave", "leaving")):
            return "exit"
        if "reflect" in text:
            return "reflection"
        return "entry"

    def __init__(self, size, narration, duration):
        w, h = size
        self.size = size
        self.duration = max(duration, .1)
        phase = self.phase(narration)
        self.observer = phase == "observer"
        self.wide = self.observer or phase == "arc"
        self.stop = {"reflection": 3, "exit": 4}.get(phase, 2)
        self.radius = min(w * .22, h * .14)
        self.center = (w * .56, h * .47)
        self.base = Image.new("RGB", size, "#0b1c2b")
        d = ImageDraw.Draw(self.base)
        if not self.wide:
            x, y = self.center
            r = self.radius
            d.ellipse((x-r,y-r,x+r,y+r), fill="#123c56", outline="#70bbce", width=4)
            d.arc((x-r+9,y-r+9,x+r-9,y+r-9), 205, 258, fill="#8ed4df", width=3)
        self.paths = [(color, [(self.center[0]+px*self.radius,self.center[1]+py*self.radius)
                               for px,py in path_for(n)]) for color,n in SPECTRUM]

    def frame(self, t):
        im = self.base.copy()
        d = ImageDraw.Draw(im)
        w,h = self.size
        progress = min(1, max(0, t / self.duration) / .78)
        if self.observer:
            geometry=sky_geometry(self.size)
            sun,eye=geometry["sun"],geometry["eye"]
            for (color,_),drop in zip(SPECTRUM,geometry["drops"]):
                for a,b,f,ink in (((sun[0],drop[1]),drop,min(1,progress*2),"#ffdb86"),
                                   (drop,eye,max(0,progress*2-1),color)):
                    if f:
                        tip=tuple(u+(v-u)*f for u,v in zip(a,b))
                        d.line([a,tip],fill=ink,width=max(1,round(w*.002)))
            sx,sy=sun
            radius=w*.033
            d.ellipse((sx-radius,sy-radius,sx+radius,sy+radius),fill="#ffdb86")
            for x,y in geometry["drops"]:
                d.ellipse((x-3,y-3,x+3,y+3),fill="#82d3e6")
            x,y=eye
            d.ellipse((x-12,y-7,x+12,y+7),outline="white",width=2)
            d.ellipse((x-3,y-3,x+3,y+3),fill="white")
            for (x,y),label in ((sun,"Sun"),(drop,"Rain"),(eye,"Observer")):
                d.text((x,y+radius+8),label,font=font(round(w*.026)),fill="#dcebef",anchor="mt")
            blend=min(1,max(0,(t/self.duration-.65)/.25))
            if blend:
                sky=self.base.copy()
                arcs=ImageDraw.Draw(sky)
                for (color,_),(cx,cy,r) in zip(SPECTRUM,geometry["arcs_center_x_y_radius"]):
                    arcs.arc((cx-r,cy-r,cx+r,cy+r),180,360,fill=color,width=max(2,round(w*.009)))
                im=Image.blend(im,sky,blend)
            return im
        if self.wide:
            # Sky view, facing away from the Sun. Bands sweep onto the same arc.
            for (color,_),(cx,cy,r) in zip(SPECTRUM,sky_geometry(self.size)["arcs_center_x_y_radius"]):
                d.arc((cx-r,cy-r,cx+r,cy+r),180,180+180*progress,fill=color,width=max(2,round(w*.009)))
            return im
        for color,points in self.paths:
            for j in range(self.stop):
                a,b = points[j:j+2]
                fraction = min(1,max(0,progress*self.stop-j))
                if not fraction:
                    break
                tip = tuple(u+(v-u)*fraction for u,v in zip(a,b))
                d.line([a,tip],fill="#fff4db" if j==0 else color,width=max(2,round(w*.0028)))
                if fraction < 1:
                    x,y=tip
                    d.ellipse((x-4,y-4,x+4,y+4),fill=color)
        return im
