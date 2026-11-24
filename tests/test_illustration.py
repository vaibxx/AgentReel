from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from agentreel import illustrate, stock
from agentreel.illustration import Illustration
from agentreel.models import Brief, Shot


def shot(i=0, mode='illustration', prompt='a glowing crystal cave with a distant waterfall'):
    return SimpleNamespace(beat_index=i, visual_mode=mode, image_prompt=prompt, stock_query='cave', stock_alternates=[])


def test_no_backend_keeps_animated_fallback(monkeypatch, tmp_path):
    from agentreel import imagegen
    monkeypatch.setattr(imagegen, 'order', lambda requested=None: [])
    events = []
    ctx = SimpleNamespace(path=tmp_path, seed=42, emit=lambda *a, **k: events.append(a[1]), check=lambda: None)
    board = SimpleNamespace(shots=[shot()], theme='midnight')
    assert illustrate.collect(Brief(topic='caves', mode='general'), board, ctx, []) == ([], [])
    assert events == ['illustration_unavailable']


def test_disabled_or_supplied_media_never_paints(monkeypatch, tmp_path):
    monkeypatch.setattr(illustrate, 'Painter', lambda where: pytest.fail('must not load the model'))
    monkeypatch.setattr(illustrate, 'device', lambda: 'cuda')
    ctx = SimpleNamespace(path=tmp_path, seed=42, emit=lambda *a, **k: None, check=lambda: None)
    board = SimpleNamespace(shots=[shot()], theme='midnight')
    assert illustrate.collect(Brief(topic='caves', mode='general'), board, ctx, [], 'none') == ([], [])
    assert illustrate.collect(Brief(topic='caves', mode='general'), board, ctx, [{'beat_index': 0, 'path': 'x.png'}]) == ([], [])


def test_unsafe_prompts_are_refused_and_style_is_added():
    assert illustrate.unsafe('portrait of a celebrity on a stage') == ['celebrity', 'portrait']
    assert illustrate.unsafe('a quiet forest at dawn') == []
    prompt = illustrate.compose('a quiet forest at dawn.', 'forest')
    assert prompt.startswith('a quiet forest at dawn,') and 'green' in prompt and 'no text' in prompt


def test_parallax_moves_near_pixels_more_than_far(tmp_path):
    w, h = 120, 200
    rng = np.random.default_rng(1)
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(tmp_path / 'still.png')
    depth = np.zeros((h, w), dtype=np.uint8); depth[:, w // 2:] = 255  # right half is near
    Image.fromarray(depth).save(tmp_path / 'depth.png')
    item = {'path': str(tmp_path / 'still.png'), 'depth': str(tmp_path / 'depth.png'), 'beat_index': 0}
    view = Illustration(item, (w, h), 'track_right', 4)
    a = np.asarray(view.frame(0)).astype(float); b = np.asarray(view.frame(4)).astype(float)
    far = np.abs(a[:, 10:w // 2 - 10] - b[:, 10:w // 2 - 10]).mean()
    near = np.abs(a[:, w // 2 + 10:-10] - b[:, w // 2 + 10:-10]).mean()
    assert near > far > 0 and view.frame(2).size == (w, h)


def test_illustration_shots_skip_stock(monkeypatch, tmp_path):
    monkeypatch.setenv('PIXABAY_API_KEY', 'test')
    monkeypatch.setattr(stock, 'search', lambda q: pytest.fail('illustration shots must not search'))
    ctx = SimpleNamespace(path=tmp_path, artifact=lambda *a: None, emit=lambda *a, **k: None)
    assert stock.collect(Brief(topic='caves', mode='general'), SimpleNamespace(shots=[shot()]), ctx, []) == []


def test_shot_accepts_illustration_mode():
    s = Shot.model_validate(dict(beat_index=0, visual='orbit', visual_description='d', label='l', accent='blue',
                                 visual_mode='illustration', image_prompt='a vast desert at dusk'))
    assert s.visual_mode == 'illustration' and s.image_prompt
