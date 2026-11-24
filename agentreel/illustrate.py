"""Local AI illustration stills for shots whose storyboard asks for them.

SD-Turbo paints a detailed still from the storyboarder's image prompt; Depth-Anything
estimates its depth so the renderer can move near and far layers separately (2.5D
parallax). Both run locally on the GPU: no API, no cost. Every image is seeded,
hashed, logged and licence-recorded. Without a usable GPU the shot keeps its animated
scene, so a run never depends on this module.
"""
import gc
import os
import re
import time
from pathlib import Path
from PIL import Image
from .runtime import sha
from .stock import BLOCKED, words

SD_REPO = 'stabilityai/sd-turbo'
DEPTH_REPO = 'depth-anything/Depth-Anything-V2-Small-hf'
SD_LICENCE = 'Stability AI Community License, July 2024 (stabilityai/sd-turbo): free for research and non-commercial use; AI-generated image'
DEPTH_LICENCE = 'Apache-2.0 (depth-anything/Depth-Anything-V2-Small-hf); derived depth map'
# Base pass at the model's native scale, then a light img2img pass at higher
# resolution adds real detail instead of only upscaling.
BASE = {'vertical': (512, 912), 'horizontal': (912, 512)}
# Sized to stay inside a 4 GB GPU; larger spills to shared memory and slows sharply.
REFINE = {'vertical': (640, 1136), 'horizontal': (1136, 640)}
# CLIP reads only 77 tokens: the subject comes first and the style tags stay short.
STYLE = 'detailed digital illustration, cinematic lighting, depth of field, sharp focus'
# One colour mood per film keeps generated stills consistent with the chosen theme.
MOOD = {'midnight': 'deep blue night tones', 'ocean': 'teal aqua tones', 'sunset': 'golden hour tones',
        'forest': 'lush green tones', 'neon': 'neon magenta cyan tones', 'paper': 'bright pastel tones',
        'ember': 'fiery orange tones', 'arctic': 'icy blue white tones'}
SAFE = 'no text'
# Requests that must never be painted, whatever the storyboard says.
UNSAFE = BLOCKED | {'portrait', 'selfie', 'celebrity', 'president', 'minister', 'victim', 'corpse', 'injury', 'logo', 'trademark'}


def model_dir():
    # The home cache, not AppData: packaged apps on Windows silently redirect AppData writes,
    # which would hide the weights from a normal terminal and force a re-download.
    default = Path.home() / '.cache' / 'agentreel' / 'models'
    return Path(os.getenv('AGENTREEL_MODEL_DIR') or default)


def device():
    """'cuda' when a GPU is usable, 'cpu' only if explicitly allowed, else None."""
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available(): return 'cuda'
    return 'cpu' if os.getenv('ALLOW_CPU_ILLUSTRATION', 'false').lower() == 'true' else None


def unsafe(prompt):
    return sorted(set(words(prompt)) & UNSAFE)


def compose(prompt, theme):
    prompt = re.sub(r'\s+', ' ', prompt).strip().rstrip('.')
    return f'{prompt}, {STYLE}, {MOOD.get(theme, MOOD["midnight"])}, {SAFE}'


class Painter:
    def __init__(self, where):
        import torch
        from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
        self.torch = torch; self.where = where
        dtype = torch.float16 if where == 'cuda' else torch.float32
        kwargs = dict(torch_dtype=dtype, cache_dir=str(model_dir()))
        if where == 'cuda': kwargs['variant'] = 'fp16'
        self.text = AutoPipelineForText2Image.from_pretrained(SD_REPO, **kwargs).to(where)
        self.text.set_progress_bar_config(disable=True)
        # The refiner shares every weight with the base pipeline; no extra memory.
        self.image = AutoPipelineForImage2Image.from_pipe(self.text)
        self.image.set_progress_bar_config(disable=True)
        if where == 'cuda':
            self.text.enable_attention_slicing(); self.text.vae.enable_tiling()

    def paint(self, prompt, seed, orientation):
        g = self.torch.Generator(self.where).manual_seed(seed)
        w, h = BASE[orientation]
        base = self.text(prompt=prompt, width=w, height=h, num_inference_steps=2, guidance_scale=0.0, generator=g).images[0]
        try:
            # SD-Turbo img2img needs steps * strength >= 1.
            big = base.resize(REFINE[orientation], Image.Resampling.LANCZOS)
            return self.image(prompt=prompt, image=big, num_inference_steps=4, strength=.35, guidance_scale=0.0,
                              generator=self.torch.Generator(self.where).manual_seed(seed)).images[0]
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            return base

    def close(self):
        del self.text, self.image
        gc.collect()
        if self.where == 'cuda': self.torch.cuda.empty_cache()


def depth_device():
    """'cuda' or 'cpu' when torch + transformers can estimate depth here, else None."""
    try:
        import torch, transformers  # noqa: F401
    except ImportError:
        return None
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def estimate_depth(paths, where):
    from transformers import pipeline
    estimator = pipeline('depth-estimation', model=DEPTH_REPO, device=0 if where == 'cuda' else -1,
                         model_kwargs={'cache_dir': str(model_dir())})
    maps = []
    for path in paths:
        # Larger values are nearer the camera for Depth-Anything.
        depth = estimator(Image.open(path).convert('RGB'))['depth'].convert('L')
        target = path.with_name(path.stem + '_depth.png'); depth.save(target); maps.append(target)
    del estimator; gc.collect()
    return maps


def jobs(board, existing):
    """(shot, prompts) to paint: illustration shots get up to three angles; a footage shot
    whose stock search found nothing gets one painted frame instead of a plain diagram."""
    taken = {int(m['beat_index']) for m in existing if isinstance(m, dict) and m.get('beat_index') not in (None, '')}
    out = []
    for shot in board.shots:
        prompt = (getattr(shot, 'image_prompt', '') or '').strip()
        if shot.beat_index in taken or not prompt: continue
        mode = getattr(shot, 'visual_mode', '')
        if mode == 'illustration':
            extra = [p.strip() for p in (getattr(shot, 'extra_image_prompts', None) or []) if p and p.strip()]
            out.append((shot, [prompt, *extra[:2]]))
        elif mode == 'footage' and os.getenv('IMAGE_FOR_MISSING_FOOTAGE', 'true').lower() == 'true':
            out.append((shot, [prompt]))
    return out


def collect(brief, board, ctx, existing, enabled='auto'):
    """Paint illustration shots (and footage shots stock could not fill) with the best
    available image backend. Returns (one media record per beat, extra manifest rows)."""
    from . import imagegen
    work = jobs(board, existing)
    if enabled == 'none' or not work: return [], []
    names = imagegen.order()
    if not names:
        ctx.emit('asset_generator', 'illustration_unavailable', inputs={'reason': 'no image backend (no Cloudflare key, no gradio_client, no CUDA GPU)',
                                                                         'shots': [s.beat_index for s, _ in work]})
        return [], []
    orientation = 'vertical' if brief.size[1] > brief.size[0] else 'horizontal'
    size = imagegen.SIZES[orientation]
    theme = getattr(board, 'theme', 'midnight')
    folder = ctx.path / 'assets'; folder.mkdir(exist_ok=True)
    limit = int(os.getenv('IMAGE_MAX_PER_RUN', '24'))
    chain = imagegen.Chain(names, ctx.emit)
    painted = {}; count = 0
    try:
        for shot, prompts in work:
            for part, raw in enumerate(prompts):
                if count >= limit: break
                blocked = unsafe(raw)
                if blocked:
                    ctx.emit('asset_generator', 'illustration_refused', inputs={'beat_index': shot.beat_index, 'blocked_terms': blocked}); continue
                prompt = compose(raw, theme)
                # Seeds derive from the run seed, so a re-run asks for the same images.
                seed = ctx.seed * 1000 + shot.beat_index * 10 + part
                ctx.check(); t = time.monotonic()
                try:
                    image, backend = chain.paint(prompt, seed, size)
                except Exception:
                    if not chain.names: break
                    continue
                path = folder / f'illustration_{shot.beat_index:02}_{part}.png'; image.save(path); count += 1
                ctx.emit('asset_generator', 'image_generation', model=backend, cost_inr=0, latency_s=round(time.monotonic() - t, 3),
                         inputs={'beat_index': shot.beat_index, 'part': part, 'prompt': prompt, 'seed': seed, 'size': list(image.size), 'backend': backend},
                         output_path=str(path.relative_to(ctx.path)), output_sha256=sha(path))
                painted.setdefault(shot.beat_index, []).append(dict(path=str(path), prompt=prompt, seed=seed, backend=backend))
            if not chain.names: break
    finally:
        chain.close()
    if not painted: return [], []
    where = depth_device(); paths = [Path(p['path']) for parts in painted.values() for p in parts]
    depths = {}
    if where:
        t = time.monotonic()
        try:
            depths = dict(zip(map(str, paths), map(str, estimate_depth(paths, where))))
            ctx.emit('asset_generator', 'depth_anything', model=DEPTH_REPO, latency_s=round(time.monotonic() - t, 3),
                     inputs={'images': len(paths), 'device': where})
        except Exception as exc:
            # Without depth the still still moves; it simply loses its parallax layers.
            ctx.emit('asset_generator', 'depth_anything', 'error', model=DEPTH_REPO, error_type=type(exc).__name__)
    from .imagegen import LICENCES
    records = []; extra = []
    for index, parts in painted.items():
        for p in parts: p['depth'] = depths.get(p['path'], '')
        first = parts[0]; first_path = Path(first['path'])
        records.append(dict(path=first['path'], file='assets/' + first_path.name, kind='generated', ai_generated=True,
                            source=f'Generated with the {first["backend"]} image backend, seed {first["seed"]}', licence=LICENCES[first['backend']],
                            sha256=sha(first_path), beat_index=index, description=first['prompt'], start_s=0,
                            identifiable_people='false', depth=first['depth'], parts=parts))
        for p in parts[1:]:
            extra.append(dict(file='assets/' + Path(p['path']).name, kind='generated', source=f'Generated with the {p["backend"]} image backend, seed {p["seed"]}',
                              licence=LICENCES[p['backend']], sha256=sha(p['path']), beat_index=index, description=p['prompt'], identifiable_people='false'))
        # Depth maps are recorded but never offered to the renderer as shot media.
        for p in parts:
            if p['depth']:
                extra.append(dict(file='assets/' + Path(p['depth']).name, kind='generated', source=f'Depth estimated with {DEPTH_REPO}',
                                  licence=DEPTH_LICENCE, sha256=sha(p['depth']), description='depth map for ' + Path(p['path']).name, identifiable_people='false'))
    ctx.artifact('illustrations.json', [{k: v for k, v in r.items() if k != 'path'} for r in records + extra], 'asset_generator')
    return records, extra
