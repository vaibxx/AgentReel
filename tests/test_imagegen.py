"""Image backends, multi-angle illustration jobs and the multi-part renderer, without network."""
import base64
import io
import shutil
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from agentreel import imagegen, illustrate, animate
from agentreel.models import Brief


def shot(i, mode, prompt='a misty pine forest at dawn', extra=()):
    return SimpleNamespace(beat_index=i, visual_mode=mode, image_prompt=prompt, extra_image_prompts=list(extra),
                           motion_prompt='mist drifts, slow push in', stock_query='forest', stock_alternates=[])


class Fake:
    name = 'fake'
    calls = []
    def __init__(self): pass
    def paint(self, prompt, seed, size):
        Fake.calls.append((prompt, seed, size)); return Image.new('RGB', size, (seed % 255, 90, 160))
    def close(self): pass


def ctx_for(tmp_path, events=None, artifacts=None):
    events = [] if events is None else events
    return SimpleNamespace(path=tmp_path, seed=42, check=lambda: None, emit=lambda *a, **k: events.append((a[1], k)),
                           artifact=lambda name, data, agent: (artifacts if artifacts is not None else {}).update({name: data}))


def test_jobs_cover_angles_and_unfilled_footage(monkeypatch):
    board = SimpleNamespace(shots=[shot(0, 'illustration', extra=['close-up of a pine cone', 'wide valley view', 'ignored third']),
                                   shot(1, 'footage'), shot(2, 'footage'), shot(3, 'animation'), shot(4, 'illustration', prompt='')])
    jobs = illustrate.jobs(board, [{'beat_index': 2, 'path': 'stock.mp4'}])
    assert [(s.beat_index, len(p)) for s, p in jobs] == [(0, 3), (1, 1)]
    monkeypatch.setenv('IMAGE_FOR_MISSING_FOOTAGE', 'false')
    assert [s.beat_index for s, _ in illustrate.jobs(board, [])] == [0]


def test_collect_records_parts_seeds_and_licences(monkeypatch, tmp_path):
    monkeypatch.setitem(imagegen.BACKENDS, 'fake', Fake)
    monkeypatch.setitem(imagegen.LICENCES, 'fake', 'test licence')
    monkeypatch.setattr(imagegen, 'order', lambda requested=None: ['fake'])
    monkeypatch.setattr(illustrate, 'depth_device', lambda: None)
    Fake.calls = []
    artifacts = {}
    board = SimpleNamespace(shots=[shot(0, 'illustration', extra=['close-up of bark']), shot(1, 'footage')], theme='forest')
    records, extra = illustrate.collect(Brief(topic='forests', mode='general'), board, ctx_for(tmp_path, artifacts=artifacts), [])
    assert [len(r['parts']) for r in records] == [2, 1]
    assert {c[2] for c in Fake.calls} == {(768, 1344)} and len({c[1] for c in Fake.calls}) == 3
    assert records[0]['licence'] == 'test licence' and extra[0]['file'].endswith('illustration_00_1.png')
    assert 'illustrations.json' in artifacts


def test_chain_moves_on_after_quota(monkeypatch):
    class Empty(Fake):
        name = 'empty'
        def paint(self, *a): raise imagegen.QuotaError('daily quota exceeded')
    monkeypatch.setitem(imagegen.BACKENDS, 'empty', Empty)
    monkeypatch.setitem(imagegen.BACKENDS, 'fake', Fake)
    events = []
    chain = imagegen.Chain(['empty', 'fake'], lambda *a, **k: events.append(a[1]))
    image, name = chain.paint('a lake', 1, (64, 112))
    assert name == 'fake' and image.size == (64, 112) and 'image_backend_exhausted' in events


def test_cloudflare_sends_documented_multipart_and_decodes(monkeypatch):
    sent = {}
    buf = io.BytesIO(); Image.new('RGB', (8, 14), 'red').save(buf, 'JPEG')
    class Reply:
        status_code = 200; text = ''
        def json(self): return {'success': True, 'result': {'image': base64.b64encode(buf.getvalue()).decode()}}
    def post(url, headers=None, files=None, json=None, timeout=None):
        sent.update(url=url, headers=headers, files=files, json=json); return Reply()
    import requests
    monkeypatch.setattr(requests, 'post', post)
    painter = imagegen.Cloudflare('acct123', 'tok456')
    image = painter.paint('a lighthouse in fog', 7, (768, 1344))
    assert sent['url'].endswith('/accounts/acct123/ai/run/@cf/black-forest-labs/flux-2-klein-4b')
    assert sent['headers'] == {'Authorization': 'Bearer tok456'}
    assert {k: v[1] for k, v in sent['files'].items()} == {'prompt': 'a lighthouse in fog', 'width': '768', 'height': '1344', 'seed': '7'}
    assert image.size == (8, 14)


def test_cloudflare_quota_reply_becomes_quota_error(monkeypatch):
    import requests
    monkeypatch.setattr(requests, 'post', lambda *a, **k: SimpleNamespace(status_code=429, text='Too Many Requests', json=lambda: {}))
    with pytest.raises(imagegen.QuotaError):
        imagegen.Cloudflare('a', 'b').paint('x', 1, (256, 256))


def test_order_prefers_cloudflare_then_space(monkeypatch):
    monkeypatch.setenv('IMAGE_BACKEND', 'auto')
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'a'); monkeypatch.setenv('CLOUDFLARE_API_TOKEN', 'b')
    monkeypatch.setattr(illustrate, 'device', lambda: None)
    assert imagegen.order()[:2] == ['cloudflare', 'hf']
    assert imagegen.order('none') == [] and imagegen.order('local') == ['local']


def make_clip(path, seconds=1.0):
    ff = shutil.which('ffmpeg')
    if not ff: pytest.skip('FFmpeg not installed')
    subprocess.run([ff, '-y', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=96x160:rate=16', '-t', str(seconds),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True)
    return path


def test_sequence_cuts_between_angles_and_labels_parts(tmp_path):
    from agentreel.motion import Sequence
    rng = np.random.default_rng(3)
    stills = []
    for i in range(2):
        p = tmp_path / f's{i}.png'; Image.fromarray(rng.integers(0, 255, (160, 96, 3), dtype=np.uint8)).save(p); stills.append(str(p))
    clip = make_clip(tmp_path / 'c.mp4')
    record = {'beat_index': 0, 'path': stills[0], 'parts': [{'video': str(clip), 'clip_s': 1.0}, {'path': stills[0]}, {'path': stills[1]}]}
    seq = Sequence(record, (96, 160), 30, 'push_in', 6.0)
    try:
        assert abs(sum(seq.durations) - 6.0) < 1e-6 and seq.durations[0] == pytest.approx(2.0)
        assert seq.label(0.5) == 'AI-GENERATED VIDEO' and seq.label(5.5) == 'AI-GENERATED IMAGE'
        frames = [np.asarray(seq.frame(t)).astype(float) for t in (0.2, 3.0, 5.5)]
        assert all(f.shape == (160, 96, 3) for f in frames) and np.abs(frames[1] - frames[2]).mean() > 5
    finally:
        seq.close()


def test_space_video_animates_first_part_and_stops_on_quota(monkeypatch, tmp_path):
    (tmp_path / 'assets').mkdir()
    stills = []
    for i in range(2):
        p = tmp_path / 'assets' / f'illustration_0{i}_0.png'; Image.new('RGB', (96, 160), 'blue').save(p); stills.append(p)
    source = make_clip(tmp_path / 'remote.mp4')

    class FakeSpace:
        name = 'hf'; where = 'remote'; calls = 0
        def clip(self, image, prompt, seconds, seed, steps=None):
            FakeSpace.calls += 1
            if FakeSpace.calls > 1: raise imagegen.QuotaError('GPU quota exceeded')
            return str(source)
        def close(self): pass
    monkeypatch.setattr(animate, 'vram_gb', lambda: 0.0)
    records = [dict(path=str(s), beat_index=i, licence='img licence', parts=[{'path': str(s)}]) for i, s in enumerate(stills)]
    events = []
    board = SimpleNamespace(shots=[shot(0, 'illustration'), shot(1, 'illustration')])
    extra = animate.collect(SimpleNamespace(size=(1080, 1920)), board, ctx_for(tmp_path, events), records, {0: 5, 1: 5}, animator=FakeSpace())
    assert records[0]['parts'][0]['video'].endswith('video_00.mp4') and records[0]['parts'][0]['clip_s'] > 0.5
    assert 'video' not in records[1]['parts'][0]
    assert any(name == 'video_quota_exhausted' for name, _ in events)
    assert extra[0]['file'] == 'assets/video_00.mp4' and 'img licence' in extra[0]['licence']
