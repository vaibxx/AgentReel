"""Reusable, topic-independent icon sprites for the scene graph.

Every shape is drawn procedurally at 384 px from one chosen colour, then given the
same bevel lighting so a scene reads as one consistent illustrated world. Shapes are
generic building blocks (a gear, a shield, a flask), never depictions of real people,
brands or places.
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageColor

N = 384
C = N / 2
# Shapes that already carry their own lighting or are meant to be see-through.
UNLIT = {'sphere', 'wave', 'disc'}
# Rotating these over time reads as natural, never as a misleading action.
SPIN = {'gear': 22, 'atom': 30, 'sun': 8, 'virus': 10}
# Squash horizontally to suggest a coin turning.
FLIP = {'coin'}


def tones(color):
    dark = tuple(round(c * .42) for c in color)
    light = tuple(min(255, round(c * .65 + 85)) for c in color)
    return dark, light


def ball(d, x, y, r, color):
    """Small lit ball for atoms, electrons, bubbles and rivets."""
    d.ellipse((x - r, y - r, x + r, y + r), fill=tuple(round(c * .55) for c in color))
    d.ellipse((x - r * .92, y - r * .96, x + r * .76, y + r * .68), fill=color)
    d.ellipse((x - r * .55, y - r * .7, x - r * .05, y - r * .2), fill=tuple(min(255, round(c * .6 + 110)) for c in color))


def teardrop(cx, top, height, width, power=1.5, steps=90):
    pts = []
    for i in range(steps + 1):
        t = i / steps * math.tau
        pts.append((cx + width * math.sin(t) * abs(math.sin(t / 2)) ** power, top + height * (1 - math.cos(t)) / 2))
    return pts


def star_points(cx, cy, outer, inner, n=5, phase=-math.pi / 2):
    return [(cx + (outer if k % 2 == 0 else inner) * math.cos(phase + k * math.pi / n),
             cy + (outer if k % 2 == 0 else inner) * math.sin(phase + k * math.pi / n)) for k in range(2 * n)]


def ellipse_points(cx, cy, rx, ry, angle, steps=90):
    a = math.radians(angle)
    return [(cx + rx * math.cos(t) * math.cos(a) - ry * math.sin(t) * math.sin(a),
             cy + rx * math.cos(t) * math.sin(a) + ry * math.sin(t) * math.cos(a))
            for t in (i / steps * math.tau for i in range(steps + 1))]


def lit(im):
    """Shared bevel: light from the top-left, a bright upper rim and a darker lower rim."""
    arr = np.asarray(im).astype(np.float32)
    alpha = arr[..., 3] / 255
    yy, xx = np.mgrid[0:N, 0:N]
    arr[..., :3] *= (1.16 - .42 * (xx + yy) / (2 * N))[..., None]
    upper = np.clip(alpha - np.roll(alpha, (7, 7), axis=(0, 1)), 0, 1)
    lower = np.clip(alpha - np.roll(alpha, (-7, -7), axis=(0, 1)), 0, 1)
    arr[..., :3] = arr[..., :3] * (1 - .38 * lower[..., None]) + 85 * upper[..., None]
    arr[..., :3] = np.clip(arr[..., :3], 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def draw(shape, hex_color):
    color = ImageColor.getrgb(hex_color)
    dark, light = tones(color)
    im = Image.new('RGBA', (N, N)); d = ImageDraw.Draw(im)
    clear = (0, 0, 0, 0)
    if shape == 'sphere':
        yy, xx = np.mgrid[-1:1:complex(N), -1:1:complex(N)]
        rr = xx * xx + yy * yy; z = np.sqrt(np.maximum(0, 1 - rr))
        illumination = np.clip(-xx * .35 - yy * .5 + z * .8, .06, 1)
        specular = np.maximum(0, -xx * .25 - yy * .35 + z * .9) ** 36
        arr = np.zeros((N, N, 4), dtype=np.uint8)
        for c in range(3): arr[:, :, c] = np.clip(color[c] * (.16 + .84 * illumination) + specular * 55, 0, 255)
        arr[:, :, 3] = (rr <= .94).astype(np.uint8) * 255
        return Image.fromarray(arr)
    if shape == 'box':
        d.polygon([(60, 95), (265, 60), (330, 120), (120, 160)], fill=light)
        d.polygon([(60, 95), (120, 160), (120, 325), (60, 255)], fill=dark)
        d.polygon([(120, 160), (330, 120), (330, 285), (120, 325)], fill=color)
    elif shape == 'leaf':
        pts = [(60 + 270 * i / 60, 200 - 120 * math.sin(i / 60 * math.pi)) for i in range(61)]
        pts += [(60 + 270 * i / 60, 200 + 85 * math.sin(i / 60 * math.pi)) for i in range(60, -1, -1)]
        d.polygon(pts, fill=color); d.line((55, 215, 325, 195), fill=light, width=8)
        for x in range(110, 300, 45):
            d.line((x, 205, x + 18, 140), fill=light, width=4); d.line((x, 205, x + 18, 260), fill=dark, width=4)
    elif shape == 'cloud':
        for x, y, r in ((100, 220, 60), (175, 170, 80), (265, 210, 70)):
            d.ellipse((x - r, y - r, x + r, y + r), fill=color)
        d.rounded_rectangle((70, 205, 310, 285), radius=35, fill=color)
    elif shape == 'person':
        d.ellipse((150, 30, 230, 110), fill=light)
        d.rounded_rectangle((125, 120, 255, 265), radius=35, fill=color)
        for pts in ((140, 245, 120, 355), (240, 245, 260, 355), (130, 145, 70, 245), (250, 145, 310, 245)):
            d.line(pts, fill=color, width=30)
    elif shape == 'building':
        d.rectangle((65, 75, 305, 330), fill=color); d.polygon([(305, 75), (340, 95), (340, 330), (305, 330)], fill=dark)
        for x in range(90, 290, 55):
            for y in range(100, 270, 55): d.rectangle((x, y, x + 28, y + 28), fill=light)
        d.rectangle((165, 270, 210, 330), fill=dark)
    elif shape == 'vessel':
        d.rounded_rectangle((90, 50, 290, 330), radius=35, fill=dark, outline=light, width=8)
        d.rectangle((100, 195, 280, 290), fill=color); d.ellipse((100, 177, 280, 213), fill=light)
    elif shape == 'ring':
        d.ellipse((35, 95, 350, 290), outline=dark, width=35); d.arc((35, 80, 350, 275), 180, 360, fill=light, width=25)
    elif shape == 'molecule':
        atoms = [(192, 192, 62), (92, 128, 44), (292, 120, 48), (120, 290, 40), (280, 282, 44)]
        for x, y, _ in atoms[1:]: d.line((192, 192, x, y), fill=light, width=14)
        for i, (x, y, r) in enumerate(atoms): ball(d, x, y, r, color if i % 2 == 0 else light)
    elif shape == 'wave':
        pts = [(20 + i * 344 / 120, 192 - 95 * math.sin(i / 120 * 4 * math.pi)) for i in range(121)]
        d.line(pts, fill=dark, width=34, joint='curve'); d.line(pts, fill=color, width=18, joint='curve'); d.line(pts, fill=light, width=6, joint='curve')
    elif shape == 'disc':
        d.ellipse((20, 20, 364, 364), fill=(*color, 120), outline=light, width=10)
        d.ellipse((70, 60, 200, 150), fill=(*light, 70))
    elif shape == 'gear':
        teeth = 12; pts = []
        for k in range(teeth * 4):
            a = k / (teeth * 4) * math.tau; r = 172 if k % 4 in (1, 2) else 138
            pts.append((C + r * math.cos(a), C + r * math.sin(a)))
        d.polygon(pts, fill=color); d.ellipse((C - 92, C - 92, C + 92, C + 92), outline=dark, width=14)
        d.ellipse((C - 42, C - 42, C + 42, C + 42), fill=clear)
    elif shape == 'bolt':
        d.polygon([(232, 18), (86, 214), (178, 214), (136, 366), (300, 146), (206, 146), (262, 18)], fill=color)
        d.line([(232, 18), (86, 214), (178, 214)], fill=light, width=8)
    elif shape == 'flame':
        d.polygon(teardrop(C, 18, 346, 150), fill=color)
        d.polygon(teardrop(C, 150, 214, 86), fill=light)
        d.polygon(teardrop(C, 240, 124, 40), fill=(255, 250, 225))
    elif shape == 'star':
        d.polygon(star_points(C, 200, 178, 74), fill=color)
        d.polygon(star_points(C, 200, 96, 40), fill=light)
    elif shape == 'heart':
        pts = []
        for i in range(181):
            t = i / 180 * math.tau
            pts.append((C + 10.6 * 16 * math.sin(t) ** 3, 178 - 10.6 * (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))))
        d.polygon(pts, fill=color); d.ellipse((95, 90, 165, 150), fill=light)
    elif shape == 'shield':
        outline = [(C, 20), (330, 68), (322, 206), (C, 364), (62, 206), (54, 68)]
        d.polygon(outline, fill=color); d.polygon([(C, 58), (296, 94), (290, 200), (C, 320)], fill=dark)
        d.line([(C, 100), (C, 290)], fill=light, width=10); d.line([(120, 170), (264, 170)], fill=light, width=10)
    elif shape == 'lock':
        d.arc((112, 32, 272, 220), 180, 360, fill=light, width=34)
        d.line((129, 126, 129, 180), fill=light, width=34); d.line((255, 126, 255, 180), fill=light, width=34)
        d.rounded_rectangle((78, 168, 306, 356), radius=28, fill=color)
        d.ellipse((168, 220, 216, 268), fill=dark); d.rectangle((182, 250, 202, 312), fill=dark)
    elif shape == 'screen':
        d.rounded_rectangle((104, 14, 280, 370), radius=34, fill=dark)
        d.rounded_rectangle((120, 52, 264, 318), radius=10, fill=light)
        for i, y in enumerate((80, 130, 180, 230, 270)):
            d.rounded_rectangle((136, y, 136 + (112 if i % 2 else 80), y + 26), radius=8, fill=color)
        d.ellipse((178, 330, 206, 358), fill=light)
    elif shape == 'document':
        d.polygon([(78, 20), (248, 20), (310, 82), (310, 364), (78, 364)], fill=(244, 244, 238))
        d.polygon([(248, 20), (310, 82), (248, 82)], fill=(196, 196, 190))
        d.rectangle((108, 52, 220, 76), fill=color)
        for y in range(118, 340, 34): d.rectangle((108, y, 280 if y % 68 else 240, y + 12), fill=dark)
    elif shape == 'bulb':
        d.ellipse((86, 18, 298, 230), fill=color)
        d.polygon([(130, 190), (254, 190), (234, 262), (150, 262)], fill=color)
        d.ellipse((128, 58, 200, 130), fill=light)
        for y in (270, 296, 322): d.rounded_rectangle((146, y, 238, y + 20), radius=8, fill=(190, 196, 205))
        d.polygon([(166, 344), (218, 344), (204, 366), (180, 366)], fill=(120, 126, 136))
    elif shape == 'clock':
        d.ellipse((24, 24, 360, 360), fill=color); d.ellipse((52, 52, 332, 332), fill=(246, 246, 240))
        for k in range(12):
            a = k / 12 * math.tau; r0 = 120 if k % 3 else 104
            d.line((C + r0 * math.cos(a), C + r0 * math.sin(a), C + 132 * math.cos(a), C + 132 * math.sin(a)), fill=dark, width=8)
        d.line((C, C, C, 92), fill=dark, width=14); d.line((C, C, 262, 230), fill=color, width=10); ball(d, C, C, 16, dark)
    elif shape == 'globe':
        d.ellipse((20, 20, 364, 364), fill=color)
        for rx in (60, 120): d.ellipse((C - rx, 20, C + rx, 364), outline=light, width=6)
        d.line((C, 20, C, 364), fill=light, width=6)
        for y, half in ((C, 172), (110, 140), (274, 140)): d.line((C - half, y, C + half, y), fill=light, width=6)
        d.ellipse((20, 20, 364, 364), outline=dark, width=10)
    elif shape == 'coin':
        d.ellipse((24, 24, 360, 360), fill=dark); d.ellipse((24, 14, 360, 350), fill=color)
        d.ellipse((70, 58, 314, 306), outline=light, width=12)
        d.polygon(star_points(C, 182, 70, 30), fill=light)
    elif shape == 'flask':
        d.rectangle((158, 18, 226, 140), fill=(220, 232, 240, 200))
        d.polygon([(158, 140), (226, 140), (340, 350), (44, 350)], fill=(220, 232, 240, 200))
        d.polygon([(118, 214), (266, 214), (340, 350), (44, 350)], fill=color)
        d.rectangle((146, 14, 238, 34), fill=(200, 210, 220))
        for x, y, r in ((170, 260, 16), (220, 300, 12), (198, 236, 9)): ball(d, x, y, r, light)
    elif shape == 'atom':
        for angle in (0, 60, 120): d.line(ellipse_points(C, C, 170, 62, angle), fill=light, width=9)
        ball(d, C, C, 46, color)
        for angle, t in ((0, .3), (60, 2.4), (120, 4.4)):
            x, y = ellipse_points(C, C, 170, 62, angle, 360)[int(t / math.tau * 360)]
            ball(d, x, y, 20, light)
    elif shape == 'dna':
        for i in range(0, 41):
            y = 18 + i * 8.7; a = i / 40 * 3 * math.pi
            x1, x2 = C + 110 * math.sin(a), C - 110 * math.sin(a)
            if i % 4 == 0: d.line((x1, y, x2, y), fill=light, width=10)
        for sign, tone in ((1, color), (-1, dark)):
            d.line([(C + sign * 110 * math.sin(i / 120 * 3 * math.pi), 18 + i * 348 / 120) for i in range(121)], fill=tone, width=22, joint='curve')
    elif shape == 'drop':
        d.polygon(teardrop(C, 14, 356, 150, power=1.1), fill=color)
        d.ellipse((122, 190, 176, 270), fill=light)
    elif shape == 'tree':
        d.polygon([(166, 230), (218, 230), (232, 366), (152, 366)], fill=(116, 78, 48))
        for x, y, r in ((C, 112, 98), (110, 190, 78), (274, 190, 78), (C, 200, 92)):
            d.ellipse((x - r, y - r, x + r, y + r), fill=color)
        d.ellipse((140, 50, 210, 110), fill=light)
    elif shape == 'rocket':
        d.polygon(teardrop(C, 312, 70, 42, power=1), fill=(255, 170, 60))
        d.polygon([(122, 250), (60, 330), (130, 314)], fill=color); d.polygon([(262, 250), (324, 330), (254, 314)], fill=color)
        d.polygon([(C, 14), (250, 120), (250, 314), (134, 314), (134, 120)], fill=(236, 240, 246))
        d.polygon([(C, 14), (250, 120), (134, 120)], fill=color)
        d.ellipse((152, 148, 232, 228), fill=dark); d.ellipse((164, 160, 220, 216), fill=light)
    elif shape == 'chip':
        for k in range(5):
            o = 90 + k * 51
            for box in ((o - 8, 26, o + 8, 70), (o - 8, 314, o + 8, 358), (26, o - 8, 70, o + 8), (314, o - 8, 358, o + 8)):
                d.rectangle(box, fill=(200, 205, 212))
        d.rounded_rectangle((62, 62, 322, 322), radius=22, fill=(30, 36, 44), outline=color, width=10)
        d.rounded_rectangle((128, 128, 256, 256), radius=12, fill=color)
    elif shape == 'sun':
        for k in range(12):
            a = k / 12 * math.tau; w = .13
            d.polygon([(C + 120 * math.cos(a - w), C + 120 * math.sin(a - w)), (C + 186 * math.cos(a), C + 186 * math.sin(a)),
                       (C + 120 * math.cos(a + w), C + 120 * math.sin(a + w))], fill=light)
        d.ellipse((C - 112, C - 112, C + 112, C + 112), fill=color)
        d.ellipse((C - 70, C - 80, C - 10, C - 20), fill=light)
    elif shape == 'mountain':
        d.polygon([(10, 350), (150, 90), (290, 350)], fill=dark)
        d.polygon([(110, 350), (262, 40), (374, 350)], fill=color)
        d.polygon([(262, 40), (310, 138), (282, 126), (262, 150), (238, 124), (214, 138)], fill=(246, 248, 252))
    elif shape == 'magnet':
        d.arc((50, 30, 334, 314), 0, 180, fill=color, width=78)
        d.rectangle((50, 60, 128, 172), fill=color); d.rectangle((256, 60, 334, 172), fill=color)
        d.rectangle((50, 20, 128, 80), fill=(214, 220, 228)); d.rectangle((256, 20, 334, 80), fill=(214, 220, 228))
    elif shape == 'crystal':
        top = [(C, 16), (300, 120), (C, 368), (84, 120)]
        d.polygon(top, fill=color); d.polygon([(C, 16), (300, 120), (C, 150)], fill=light)
        d.polygon([(C, 16), (84, 120), (C, 150)], fill=tuple(min(255, c + 30) for c in color))
        d.polygon([(84, 120), (C, 150), (C, 368)], fill=dark)
    elif shape == 'virus':
        for k in range(12):
            a = k / 12 * math.tau
            d.line((C + 100 * math.cos(a), C + 100 * math.sin(a), C + 160 * math.cos(a), C + 160 * math.sin(a)), fill=dark, width=14)
            ball(d, C + 166 * math.cos(a), C + 166 * math.sin(a), 18, light)
        d.ellipse((C - 118, C - 118, C + 118, C + 118), fill=color)
        for x, y in ((150, 150), (230, 210), (170, 240)): d.ellipse((x - 18, y - 18, x + 18, y + 18), fill=dark)
    elif shape == 'arrow':
        d.polygon([(30, 150), (220, 150), (220, 70), (362, C), (220, 314), (220, 234), (30, 234)], fill=color)
        d.line([(30, 150), (220, 150), (220, 70)], fill=light, width=8)
    elif shape == 'battery':
        d.rounded_rectangle((60, 92, 314, 292), radius=24, fill=(48, 54, 62)); d.rectangle((314, 152, 346, 232), fill=(160, 166, 174))
        for k in range(4): d.rounded_rectangle((82 + k * 56, 114, 126 + k * 56, 270), radius=8, fill=color)
    elif shape == 'cell':
        d.ellipse((14, 44, 370, 340), fill=(*color, 150), outline=light, width=10)
        d.ellipse((140, 130, 250, 240), fill=dark); d.ellipse((160, 148, 206, 190), fill=light)
        for x, y in ((80, 180), (300, 160), (250, 290), (110, 260)): d.ellipse((x - 16, y - 10, x + 16, y + 10), fill=light)
    if shape not in UNLIT:
        im = lit(im)
    return im
