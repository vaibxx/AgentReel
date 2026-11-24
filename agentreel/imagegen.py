"""Image generation backends for illustration shots, tried in order until one works.

  cloudflare  FLUX.2 [klein] 4B on Cloudflare Workers AI. Free plan: 10,000 neurons a day,
              about 100 vertical 768x1344 images. Needs CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN.
  hf          FLUX.1 [schnell] on the official Hugging Face Space, through gradio_client. Free:
              uses the daily ZeroGPU allowance (HF_TOKEN raises it from the guest quota).
  local       SD-Turbo on this machine's CUDA GPU (agentreel.illustrate.Painter).

IMAGE_BACKEND=auto tries cloudflare (if configured), then hf, then local. A quota or network
failure moves the rest of the run to the next backend, so one exhausted service never stops a
video. Nothing here is paid: no backend is used that bills a card.
"""
import base64
import io
import os
import time
from PIL import Image

SIZES = {'vertical': (768, 1344), 'horizontal': (1344, 768)}
CF_MODEL = '@cf/black-forest-labs/flux-2-klein-4b'
HF_SPACE = 'black-forest-labs/FLUX.1-schnell'
LICENCES = {
    'cloudflare': 'FLUX.2 [klein] 4B by Black Forest Labs (Apache-2.0), generated on Cloudflare Workers AI free plan; AI-generated image',
    'hf': 'FLUX.1 [schnell] by Black Forest Labs (Apache-2.0), generated on the black-forest-labs Hugging Face Space; AI-generated image',
    'local': 'Stability AI Community License (stabilityai/sd-turbo), generated locally; AI-generated image',
}


class QuotaError(RuntimeError):
    """The service refused for quota or rate reasons; later images should use another backend."""


def quota_like(message):
    text = str(message).lower()
    return any(k in text for k in ('quota', 'rate limit', 'too many requests', '429', 'exceeded', 'limit reached', 'neurons'))


class Cloudflare:
    name = 'cloudflare'

    def __init__(self, account=None, token=None, model=None):
        self.account = account or os.getenv('CLOUDFLARE_ACCOUNT_ID', '')
        self.token = token or os.getenv('CLOUDFLARE_API_TOKEN', '')
        self.model = model or os.getenv('CLOUDFLARE_IMAGE_MODEL', CF_MODEL)
        if not (self.account and self.token): raise RuntimeError('CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are not set')

    def paint(self, prompt, seed, size):
        import requests
        url = f'https://api.cloudflare.com/client/v4/accounts/{self.account}/ai/run/{self.model}'
        headers = {'Authorization': f'Bearer {self.token}'}
        w, h = size
        if 'flux-2' in self.model:
            # FLUX.2 models take multipart form fields (width/height 256-1920, steps fixed at 4).
            files = {'prompt': (None, prompt[:2048]), 'width': (None, str(w)), 'height': (None, str(h)), 'seed': (None, str(seed % 2147483647))}
            r = requests.post(url, headers=headers, files=files, timeout=(15, 120))
        else:
            # FLUX.1 schnell takes JSON and always returns a square image.
            r = requests.post(url, headers=headers, json={'prompt': prompt[:2048], 'steps': 4, 'seed': seed % 2147483647}, timeout=(15, 120))
        if r.status_code == 429 or (r.status_code >= 400 and quota_like(r.text)):
            raise QuotaError(f'Cloudflare HTTP {r.status_code}')
        if r.status_code != 200:
            raise RuntimeError(f'Cloudflare HTTP {r.status_code}: {r.text[:200]}')
        data = r.json()
        image = (data.get('result') or {}).get('image') if isinstance(data, dict) else None
        if not image: raise RuntimeError('Cloudflare returned no image: ' + str(data)[:200])
        if image.startswith('data:'): image = image.split(',', 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(image))).convert('RGB')

    def close(self): pass


class Space:
    name = 'hf'

    def __init__(self, space=None):
        from gradio_client import Client
        self.space = space or os.getenv('IMAGE_HF_SPACE', HF_SPACE)
        self.client = Client(self.space, token=os.getenv('HF_TOKEN') or None, verbose=False)

    def paint(self, prompt, seed, size):
        from gradio_client.exceptions import AppError
        w, h = (v // 32 * 32 for v in size)
        try:
            result, _ = self.client.predict(prompt[:1500], seed % 2147483647, False, w, h, 4, api_name='/infer')
        except AppError as exc:
            if quota_like(exc): raise QuotaError(f'Hugging Face Space: {str(exc)[:200]}') from None
            raise RuntimeError(f'Hugging Face Space: {str(exc)[:200]}') from None
        path = result.get('path') if isinstance(result, dict) else result
        return Image.open(path).convert('RGB')

    def close(self): pass


class Local:
    name = 'local'

    def __init__(self):
        from .illustrate import Painter, device
        where = device()
        if not where: raise RuntimeError('no CUDA GPU for local SD-Turbo')
        self.painter = Painter(where)

    def paint(self, prompt, seed, size):
        return self.painter.paint(prompt, seed, 'vertical' if size[1] > size[0] else 'horizontal')

    def close(self): self.painter.close()


BACKENDS = {'cloudflare': Cloudflare, 'hf': Space, 'local': Local}


def order(requested=None):
    """Backend names to try, best first. 'none' disables generation."""
    requested = (requested or os.getenv('IMAGE_BACKEND', 'auto')).lower()
    if requested == 'none': return []
    if requested in BACKENDS: return [requested]
    names = []
    if os.getenv('CLOUDFLARE_ACCOUNT_ID') and os.getenv('CLOUDFLARE_API_TOKEN'): names.append('cloudflare')
    try:
        import gradio_client  # noqa: F401
        if os.getenv('IMAGE_HF_DISABLED', 'false').lower() != 'true': names.append('hf')
    except ImportError:
        pass
    from .illustrate import device
    if device(): names.append('local')
    return names


def available(requested=None):
    return bool(order(requested))


class Chain:
    """Paint with the first backend that works; on quota or failure move to the next."""

    def __init__(self, names, emit=lambda *a, **k: None):
        self.names = list(names); self.emit = emit; self.current = None

    def _open(self):
        while self.names and self.current is None:
            name = self.names[0]; t = time.monotonic()
            try:
                self.current = BACKENDS[name]()
                self.emit('asset_generator', 'image_backend', inputs={'backend': name}, latency_s=round(time.monotonic() - t, 3))
            except Exception as exc:
                self.emit('asset_generator', 'image_backend', 'error', error_type=type(exc).__name__, inputs={'backend': name, 'detail': str(exc)[:200]})
                self.names.pop(0)
        return self.current

    def paint(self, prompt, seed, size):
        """Returns (image, backend name). Raises RuntimeError once every backend failed."""
        last = None
        while self._open():
            try:
                return self.current.paint(prompt, seed, size), self.current.name
            except QuotaError as exc:
                last = exc; self.emit('asset_generator', 'image_backend_exhausted', inputs={'backend': self.current.name, 'detail': str(exc)[:200]})
                self.close_current(); self.names.pop(0)
            except Exception as exc:
                last = exc; self.emit('asset_generator', 'image_generation', 'error', error_type=type(exc).__name__,
                                      inputs={'backend': self.current.name, 'detail': str(exc)[:200]})
                # One failed prompt is not a dead service: keep the backend, give up on this image.
                raise
        raise RuntimeError('No image backend available' + (f': {last}' if last else ''))

    def close_current(self):
        if self.current is not None:
            try: self.current.close()
            finally: self.current = None

    def close(self): self.close_current()
