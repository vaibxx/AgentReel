"""2.5D renderer for AI illustration stills: depth parallax, camera moves, overlays.

Near pixels (bright in the depth map) shift further than far ones as the virtual
camera moves, so a single still reads as a scene with depth rather than a zoomed photo.
"""
import math
import numpy as np
from PIL import Image, ImageOps


def ease(x):
    x = max(0., min(1., x)); return x * x * (3 - 2 * x)


class Illustration:
    # Overscan leaves room for camera travel without exposing edges.
    OVERSCAN = 1.12
    PARALLAX = .035  # fraction of width that the nearest layer travels

    def __init__(self, item, size, camera, duration, overlay=None):
        self.size = size; self.camera = camera; self.duration = max(.1, duration); self.overlay = overlay
        w, h = size; W, H = round(w * self.OVERSCAN), round(h * self.OVERSCAN)
        image = ImageOps.fit(Image.open(item['path']).convert('RGB'), (W, H), method=Image.Resampling.LANCZOS)
        self.source = np.asarray(image)
        depth_path = item.get('depth')
        if depth_path:
            depth = ImageOps.fit(Image.open(depth_path).convert('L'), (w, h), method=Image.Resampling.BILINEAR)
            self.depth = np.asarray(depth, dtype=np.float32) / 255
        else:
            self.depth = np.full((h, w), .5, dtype=np.float32)
        # Output pixel grid relative to the centre, reused every frame.
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self.dx = xx - w / 2; self.dy = yy - h / 2
        self.cx, self.cy = W / 2, H / 2
        vignette = 1 - .32 * np.clip(((xx / w - .5) ** 2 + (yy / h - .5) ** 2) * 2.4, 0, 1)
        self.vignette = vignette[..., None].astype(np.float32)
        self.sign = -1 if int(item.get('beat_index') or 0) % 2 else 1

    def frame(self, t):
        p = ease(t / self.duration); w, h = self.size
        zoom, pan_x, pan_y = 1.0, 0.0, 0.0
        if self.camera == 'push_in': zoom = 1 + .07 * p; pan_y = -.012 * p
        elif self.camera == 'track_left': pan_x = -1 + 2 * (1 - p)
        elif self.camera == 'track_right': pan_x = -1 + 2 * p
        elif self.camera == 'orbit': pan_x = math.sin(p * math.pi - math.pi / 2); zoom = 1 + .03 * p
        if self.camera == 'push_in': pan_x = self.sign * (p - .5)
        # Parallax: near layers move with the camera, far layers stay nearly still.
        shift = pan_x * self.PARALLAX * w * (.15 + .85 * self.depth)
        lift = pan_y * h * (.3 + .7 * self.depth)
        sx = np.clip(self.cx + self.dx / zoom + shift, 0, self.source.shape[1] - 1).astype(np.int32)
        sy = np.clip(self.cy + self.dy / zoom + lift, 0, self.source.shape[0] - 1).astype(np.int32)
        pixels = self.source[sy, sx].astype(np.float32) * self.vignette
        im = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).convert('RGBA')
        if self.overlay is not None: im.alpha_composite(self.overlay.frame(t))
        return im.convert('RGB')
