"""Local image-to-video: turn each painted illustration still into a real moving clip.

Two open models, both run through diffusers on an NVIDIA GPU with no API or cost:
  wan  Wan2.2 TI2V-5B (Apache-2.0). Best motion and detail; wants a ~24 GB GPU.
  ltx  LTX-Video 0.9.5 (LTXV open weights licence). Much faster; ~12-16 GB is enough.
  hf   Wan 2.2 14B on a public Hugging Face ZeroGPU Space: no local GPU, free daily quota
       (about 5 GPU-minutes with a free HF_TOKEN, 2 as a guest), so a few hero clips a run.
VIDEO_MODEL=auto uses a local model when the GPU is big enough, else the Space; when nothing
works the illustration keeps its depth-parallax still. Every clip is seeded, timed, hashed and licence-recorded,
and a time budget stops new clips before a live run can overrun.
"""
import gc
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from .runtime import sha
from .speech import executable
from .illustrate import model_dir
from .imagegen import QuotaError, quota_like

NEGATIVE = ('blurry, low quality, distorted, deformed, warped faces, extra limbs, flicker, jitter, '
            'static frame, text, watermark, logo, subtitles, oversaturated, worst quality')

BACKENDS = {
    'wan': dict(repo='Wan-AI/Wan2.2-TI2V-5B-Diffusers', fps=24, frame_step=4, max_frames=121, min_vram=20,
                # TI2V-5B is trained at 720p; both sides must divide by 32.
                size={'vertical': (704, 1280), 'horizontal': (1280, 704)}, steps=30, guidance=5.0,
                licence='Apache-2.0 (Wan-AI/Wan2.2-TI2V-5B-Diffusers); AI-generated video',
                patterns=['model_index.json', '*.md', 'scheduler/*', 'tokenizer/*', 'text_encoder/*', 'transformer/*', 'vae/*']),
    'hf': dict(repo='zerogpu-aoti/wan2-2-fp8da-aoti-faster', fps=16, min_vram=0, remote=True, steps=6,
               licence='Wan2.2 I2V-A14B (Apache-2.0) run on the zerogpu-aoti/wan2-2-fp8da-aoti-faster Hugging Face Space (free ZeroGPU quota); AI-generated video'),
    'ltx': dict(repo='Lightricks/LTX-Video-0.9.5', fps=24, frame_step=8, max_frames=161, min_vram=12,
                size={'vertical': (512, 896), 'horizontal': (896, 512)}, steps=40, guidance=3.0,
                licence='LTX-Video 0.9.5 Open Weights License (Lightricks/LTX-Video-0.9.5); AI-generated video',
                patterns=['model_index.json', '*.md', '*license*', 'scheduler/*', 'tokenizer/*', 'text_encoder/*', 'transformer/*', 'vae/*']),
}


def vram_gb():
    try:
        import torch
        if torch.cuda.is_available(): return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except ImportError:
        pass
    return 0.


def space_ready():
    if os.getenv('VIDEO_HF_DISABLED', 'false').lower() == 'true': return False
    try:
        import gradio_client  # noqa: F401
        return True
    except ImportError:
        return False


def choose(requested=None):
    """Backend name to use, or None. Local models need a GPU; the Space needs gradio_client."""
    requested = (requested or os.getenv('VIDEO_MODEL', 'auto')).lower()
    if requested == 'none': return None
    if requested == 'hf': return 'hf' if space_ready() else None
    memory = vram_gb()
    if requested in ('wan', 'ltx'): return requested if memory else None
    for name in ('wan', 'ltx'):
        if memory >= BACKENDS[name]['min_vram']: return name
    return 'hf' if space_ready() else None


class SpaceAnimator:
    """Wan 2.2 14B image-to-video on a Hugging Face ZeroGPU Space via gradio_client."""
    def __init__(self, space=None):
        from gradio_client import Client
        self.name = 'hf'; self.where = 'remote'
        self.space = space or os.getenv('VIDEO_HF_SPACE', BACKENDS['hf']['repo'])
        self.client = Client(self.space, token=os.getenv('HF_TOKEN') or None, verbose=False)

    def clip(self, image_path, prompt, seconds, seed, steps=None):
        from gradio_client import handle_file
        from gradio_client.exceptions import AppError
        try:
            video, _ = self.client.predict(handle_file(str(image_path)), prompt, steps or BACKENDS['hf']['steps'], NEGATIVE, seconds,
                                           1, 1, seed % 2147483647, False, api_name='/generate_video')
        except AppError as exc:
            if quota_like(exc): raise QuotaError(f'Hugging Face Space: {str(exc)[:200]}') from None
            raise RuntimeError(f'Hugging Face Space: {str(exc)[:200]}') from None
        return video.get('video') if isinstance(video, dict) else video

    def close(self): pass


def clip_seconds(path):
    out = subprocess.run([executable('ffprobe'), '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
                         capture_output=True, text=True)
    return float(out.stdout.strip() or 0)


def frame_count(seconds, spec):
    """Frames to generate for a beat: playback may slow up to 1.6x, so a clip need not
    cover the whole beat. Counts follow each model's k*step+1 rule."""
    wanted = seconds * spec['fps'] / 1.6
    k = max(2, min((spec['max_frames'] - 1) // spec['frame_step'], math.ceil(wanted / spec['frame_step'])))
    return k * spec['frame_step'] + 1


def motion_text(shot):
    base = (getattr(shot, 'motion_prompt', '') or '').strip()
    scene = (getattr(shot, 'image_prompt', '') or '').strip()
    # The still already fixes the look; the prompt mainly says what moves and how the camera moves.
    text = f'{base}. {scene}' if base else f'{scene}. Natural gentle motion of the subject, subtle ambient movement, slow cinematic camera push in'
    return text[:900]


class Animator:
    def __init__(self, name, where='cuda'):
        import torch
        self.torch = torch; self.name = name; self.spec = BACKENDS[name]; self.where = where
        dtype = torch.bfloat16
        kwargs = dict(torch_dtype=dtype, cache_dir=str(model_dir()))
        if name == 'wan':
            from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
            # The Wan VAE is numerically safest in float32, as the model card recommends.
            vae = AutoencoderKLWan.from_pretrained(self.spec['repo'], subfolder='vae', torch_dtype=torch.float32, cache_dir=str(model_dir()))
            self.pipe = WanImageToVideoPipeline.from_pretrained(self.spec['repo'], vae=vae, image_encoder=None, image_processor=None,
                                                                transformer_2=None, **kwargs)
        else:
            from diffusers import LTXConditionPipeline
            self.pipe = LTXConditionPipeline.from_pretrained(self.spec['repo'], **kwargs)
        self.pipe.set_progress_bar_config(disable=True)
        # Offloading keeps only the active sub-model on the GPU: slower than all-resident,
        # but it fits the 5B/2B models plus their large text encoders on 16-24 GB cards.
        if where == 'cuda': self.pipe.enable_model_cpu_offload()
        else: self.pipe.to(where)
        vae = getattr(self.pipe, 'vae', None)
        if vae is not None and hasattr(vae, 'enable_tiling'): vae.enable_tiling()

    @classmethod
    def from_pipe(cls, name, pipe, where='cpu'):
        """Wrap an already built pipeline (used by tests with tiny random weights)."""
        self = cls.__new__(cls)
        import torch
        self.torch = torch; self.name = name; self.spec = BACKENDS[name]; self.where = where; self.pipe = pipe
        return self

    def run(self, image, prompt, frames, seed, size, steps=None, extra=None):
        w, h = size
        image = ImageOps.fit(image.convert('RGB'), (w, h), method=Image.Resampling.LANCZOS)
        g = self.torch.Generator('cpu').manual_seed(seed)
        args = dict(image=image, prompt=prompt, negative_prompt=NEGATIVE, height=h, width=w, num_frames=frames,
                    num_inference_steps=steps or self.spec['steps'], guidance_scale=self.spec['guidance'],
                    generator=g, output_type='np')
        if self.name == 'ltx':
            args.update(frame_rate=self.spec['fps'], decode_timestep=.05, image_cond_noise_scale=.025)
        args.update(extra or {})
        video = self.pipe(**args).frames[0]
        video = np.asarray(video)
        if video.ndim == 4 and video.shape[1] == 3 and video.shape[-1] != 3: video = video.transpose(0, 2, 3, 1)
        if video.dtype != np.uint8: video = (np.clip(video, 0, 1) * 255).round().astype(np.uint8)
        return video

    def close(self):
        del self.pipe
        gc.collect()
        if self.where == 'cuda': self.torch.cuda.empty_cache()


def encode(frames, fps, path):
    """Write RGB frames as a high-quality H.264 clip via FFmpeg (no extra video library)."""
    n, h, w, _ = frames.shape
    proc = subprocess.Popen([executable('ffmpeg'), '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
                             '-r', str(fps), '-i', 'pipe:0', '-c:v', 'libx264', '-preset', 'medium', '-crf', '16',
                             '-pix_fmt', 'yuv420p', str(path)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = proc.communicate(frames.tobytes())
    if proc.returncode: raise RuntimeError('Clip encoder failed: ' + err.decode(errors='replace')[-600:])
    return n / fps


def collect(brief, board, ctx, painted, durations, requested=None, animator=None):
    """Animate painted illustration records in place. Returns extra manifest rows.

    Each animated record keeps its still under 'still' and gains 'path' to the clip,
    'video'=True and 'clip_s', so the renderer plays motion instead of parallax."""
    if not painted: return []
    name = animator.name if animator else choose(requested)
    if not name:
        ctx.emit('asset_generator', 'video_unavailable', inputs={'reason': f'VIDEO_MODEL={requested or os.getenv("VIDEO_MODEL", "auto")}, GPU {vram_gb():.1f} GB',
                                                                 'shots': [r['beat_index'] for r in painted]})
        return []
    spec = BACKENDS[name]; remote = spec.get('remote', False)
    orientation = 'vertical' if brief.size[1] > brief.size[0] else 'horizontal'
    size = spec.get('size', {}).get(orientation)
    # e.g. VIDEO_SIZE=480x832 trades detail for speed on a smaller card.
    if os.getenv('VIDEO_SIZE') and not remote: size = tuple(int(v) // 32 * 32 for v in os.getenv('VIDEO_SIZE').lower().split('x'))
    budget = float(os.getenv('VIDEO_TIME_BUDGET_S', '3600')); steps = int(os.getenv('VIDEO_STEPS', '0')) or None
    # The free Space quota covers a few clips a day; local GPUs are limited by the time budget.
    max_clips = int(os.getenv('VIDEO_MAX_CLIPS', '0') or 0) or (3 if remote else 99)
    shots = {s.beat_index: s for s in board.shots}
    started = time.monotonic()
    if animator is None:
        try:
            animator = SpaceAnimator() if remote else Animator(name)
        except Exception as exc:
            ctx.emit('asset_generator', 'video_unavailable', 'error', model=spec['repo'], error_type=type(exc).__name__, inputs={'detail': str(exc)[:300]})
            return []
        ctx.emit('asset_generator', 'load_model', model=spec['repo'], latency_s=round(time.monotonic() - started, 3), inputs={'backend': name, 'vram_gb': round(vram_gb(), 1)})
    # Real illustration shots first: they were designed as the visual heroes of the film.
    order = sorted(painted, key=lambda r: (getattr(shots.get(int(r['beat_index'])), 'visual_mode', '') != 'illustration', int(r['beat_index'])))
    extra = []; spent = []; made = 0
    try:
        for record in order:
            index = int(record['beat_index'])
            if made >= max_clips:
                ctx.emit('asset_generator', 'video_clip_limit', inputs={'beat_index': index, 'max_clips': max_clips}); continue
            # Stop before a clip that would push the run past its video budget.
            if spent and (time.monotonic() - started) + sum(spent) / len(spent) > budget:
                ctx.emit('asset_generator', 'video_budget_reached', inputs={'beat_index': index, 'budget_s': budget}); continue
            target = record['parts'][0] if record.get('parts') else record
            still = target['path']
            ctx.check(); t = time.monotonic()
            seed = ctx.seed * 1000 + 500 + index
            prompt = motion_text(shots.get(index))
            clip = Path(still).with_name(f'video_{index:02}.mp4')
            try:
                if remote:
                    wanted = float(os.getenv('VIDEO_HF_SECONDS', '3.5'))
                    shutil.copyfile(animator.clip(still, prompt, wanted, seed, steps), clip)
                    seconds = clip_seconds(clip); frames = round(seconds * spec['fps'])
                else:
                    frames = frame_count(durations.get(index, 5), spec)
                    seconds = encode(animator.run(Image.open(still), prompt, frames, seed, size, steps), spec['fps'], clip)
            except QuotaError as exc:
                ctx.emit('asset_generator', 'video_quota_exhausted', inputs={'beat_index': index, 'backend': name, 'detail': str(exc)[:200]})
                break
            except Exception as exc:
                ctx.emit('asset_generator', 'image_to_video', 'error', model=spec['repo'], error_type=type(exc).__name__, inputs={'beat_index': index, 'detail': str(exc)[:300]})
                if animator.where == 'cuda': animator.torch.cuda.empty_cache()
                continue
            spent.append(time.monotonic() - t); made += 1
            ctx.emit('asset_generator', 'image_to_video', model=spec['repo'], cost_inr=0, latency_s=round(spent[-1], 3),
                     inputs={'beat_index': index, 'prompt': prompt, 'seed': seed, 'frames': frames, 'size': list(size) if size else 'model default', 'backend': name},
                     output_path=str(clip.relative_to(ctx.path)), output_sha256=sha(clip))
            still_licence = record['licence']
            licence = spec['licence'] + '; first frame: ' + still_licence
            source = f'Animated with {spec["repo"]} from {Path(still).name}, seed {seed}'
            if record.get('parts'):
                # Multi-image beat: the first part becomes motion, the other angles stay stills.
                target.update(video=str(clip), clip_s=seconds); record['video'] = True
                extra.append(dict(file='assets/' + clip.name, kind='generated', source=source, licence=licence, sha256=sha(clip),
                                  beat_index=index, description=prompt, identifiable_people='false'))
            else:
                record.update(still=still, path=str(clip), file='assets/' + clip.name, video=True, clip_s=seconds,
                              source=source, licence=licence, sha256=sha(clip))
                extra.append(dict(file='assets/' + Path(still).name, kind='generated', source='Illustration still used as the first frame of ' + clip.name,
                                  licence=still_licence, sha256=sha(still), beat_index=index, identifiable_people='false'))
    finally:
        if animator is not None: animator.close()
    ctx.artifact('videos.json', [{k: v for k, v in r.items() if k not in ('path', 'still')} for r in painted if r.get('video')], 'asset_generator')
    return extra
