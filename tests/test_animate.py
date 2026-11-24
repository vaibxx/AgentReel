"""Image-to-video wrapper checks with tiny random-weight diffusers pipelines on CPU.

Configs follow diffusers' own fast tests (v0.41.0), so the real call signatures, latent
shapes and frame decoding are exercised without downloading the 25-34 GB models."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip('torch')
diffusers = pytest.importorskip('diffusers')
from agentreel import animate


def tiny_text():
    from transformers import AutoConfig, AutoTokenizer, T5EncoderModel
    try:
        config = AutoConfig.from_pretrained('hf-internal-testing/tiny-random-t5')
        tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/tiny-random-t5')
    except Exception as exc:  # offline machines skip rather than fail
        pytest.skip(f'tiny tokenizer unavailable: {type(exc).__name__}')
    return T5EncoderModel(config).eval(), tokenizer


def tiny_wan():
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline, WanTransformer3DModel
    torch.manual_seed(0)
    vae = AutoencoderKLWan(base_dim=3, z_dim=48, in_channels=12, out_channels=12, is_residual=True, patch_size=2,
                           latents_mean=[0.0] * 48, latents_std=[1.0] * 48, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                           scale_factor_spatial=16, scale_factor_temporal=4, temperal_downsample=[False, True, True])
    text_encoder, tokenizer = tiny_text()
    transformer = WanTransformer3DModel(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=48,
                                        out_channels=48, text_dim=32, freq_dim=256, ffn_dim=32, num_layers=2, cross_attn_norm=True,
                                        qk_norm='rms_norm_across_heads', rope_max_seq_len=32)
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, flow_shift=3.0)
    return WanImageToVideoPipeline(transformer=transformer, vae=vae, scheduler=scheduler, text_encoder=text_encoder, tokenizer=tokenizer,
                                   transformer_2=None, image_encoder=None, image_processor=None, boundary_ratio=None, expand_timesteps=True)


def tiny_ltx():
    from diffusers import AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXConditionPipeline, LTXVideoTransformer3DModel
    torch.manual_seed(0)
    transformer = LTXVideoTransformer3DModel(in_channels=8, out_channels=8, patch_size=1, patch_size_t=1, num_attention_heads=4,
                                             attention_head_dim=8, cross_attention_dim=32, num_layers=1, caption_channels=32)
    vae = AutoencoderKLLTXVideo(in_channels=3, out_channels=3, latent_channels=8, block_out_channels=(8, 8, 8, 8),
                                decoder_block_out_channels=(8, 8, 8, 8), layers_per_block=(1, 1, 1, 1, 1),
                                decoder_layers_per_block=(1, 1, 1, 1, 1), spatio_temporal_scaling=(True, True, False, False),
                                decoder_spatio_temporal_scaling=(True, True, False, False), decoder_inject_noise=(False,) * 5,
                                upsample_residual=(False,) * 4, upsample_factor=(1,) * 4, timestep_conditioning=False,
                                patch_size=1, patch_size_t=1, encoder_causal=True, decoder_causal=False)
    vae.use_framewise_encoding = False; vae.use_framewise_decoding = False
    text_encoder, tokenizer = tiny_text()
    return LTXConditionPipeline(transformer=transformer, vae=vae, scheduler=FlowMatchEulerDiscreteScheduler(),
                                text_encoder=text_encoder, tokenizer=tokenizer)


@pytest.mark.parametrize('name,build', [('wan', tiny_wan), ('ltx', tiny_ltx)])
def test_backend_returns_rgb_uint8_frames(name, build):
    animator = animate.Animator.from_pipe(name, build())
    frames = animate.frame_count(.5, animate.BACKENDS[name])
    video = animator.run(Image.new('RGB', (40, 60), (200, 120, 40)), 'a cat walks', frames, 7, (32, 32), steps=2,
                         extra={'max_sequence_length': 16})
    assert video.dtype == np.uint8 and video.shape[1:] == (32, 32, 3) and video.shape[0] >= 9


def test_collect_turns_still_into_clip_with_records(tmp_path, monkeypatch):
    monkeypatch.setenv('VIDEO_SIZE', '32x32')
    still = tmp_path / 'assets' / 'illustration_02.png'; still.parent.mkdir()
    Image.new('RGB', (64, 112), (30, 140, 90)).save(still)
    events, artifacts = [], {}
    ctx = SimpleNamespace(path=tmp_path, seed=42, check=lambda: None, emit=lambda *a, **k: events.append((a[1], k)),
                          artifact=lambda name, data, agent: artifacts.update({name: data}))
    record = dict(path=str(still), file='assets/illustration_02.png', beat_index=2, licence='SD licence', kind='generated', ai_generated=True)
    board = SimpleNamespace(shots=[SimpleNamespace(beat_index=2, image_prompt='a frog on a log', motion_prompt='the frog leaps')])
    animator = animate.Animator.from_pipe('wan', tiny_wan())
    real_run = animator.run
    animator.run = lambda *a, **k: real_run(*a, **{**k, 'extra': {'max_sequence_length': 16}})
    monkeypatch.setenv('VIDEO_STEPS', '2')
    extra = animate.collect(SimpleNamespace(size=(1080, 1920)), board, ctx, [record], {2: .4}, animator=animator)
    clip = tmp_path / 'assets' / 'video_02.mp4'
    assert clip.is_file() and record['video'] and record['path'] == str(clip) and record['still'] == str(still)
    assert record['clip_s'] > 0 and 'Apache-2.0' in record['licence'] and 'SD licence' in record['licence']
    assert extra[0]['file'] == 'assets/illustration_02.png'
    assert any(name == 'image_to_video' and k.get('inputs', {}).get('prompt', '').startswith('the frog leaps') for name, k in events)
    assert artifacts['videos.json'][0]['beat_index'] == 2


def test_frame_counts_follow_model_rules():
    for spec in (s for s in animate.BACKENDS.values() if not s.get('remote')):
        for seconds in (1, 4, 9, 30):
            n = animate.frame_count(seconds, spec)
            assert (n - 1) % spec['frame_step'] == 0 and n <= spec['max_frames']


def test_auto_skips_small_gpus(monkeypatch):
    monkeypatch.setenv('VIDEO_MODEL', 'auto')
    monkeypatch.setattr(animate, 'space_ready', lambda: False)
    monkeypatch.setattr(animate, 'vram_gb', lambda: 4.0)
    assert animate.choose() is None
    monkeypatch.setattr(animate, 'space_ready', lambda: True)
    assert animate.choose() == 'hf' and animate.choose('ltx') == 'ltx'
    monkeypatch.setattr(animate, 'vram_gb', lambda: 0.0)
    assert animate.choose('wan') is None and animate.choose('hf') == 'hf'
    monkeypatch.setattr(animate, 'vram_gb', lambda: 16.0)
    assert animate.choose() == 'ltx'
    monkeypatch.setattr(animate, 'vram_gb', lambda: 24.0)
    assert animate.choose() == 'wan'
    assert animate.choose('none') is None


@pytest.mark.parametrize('name,build', [('wan', tiny_wan), ('ltx', tiny_ltx)])
def test_loader_accepts_the_real_repository_layout(name, build, tmp_path, monkeypatch):
    """Save tiny weights under the real repo's model_index.json, then load with Animator."""
    from huggingface_hub import hf_hub_download
    try:
        index = json.loads(open(hf_hub_download(animate.BACKENDS[name]['repo'], 'model_index.json')).read())
    except Exception as exc:
        pytest.skip(f'model index unavailable offline: {type(exc).__name__}')
    build().save_pretrained(tmp_path)
    (tmp_path / 'model_index.json').write_text(json.dumps(index))
    monkeypatch.setitem(animate.BACKENDS[name], 'repo', str(tmp_path))
    animator = animate.Animator(name, where='cpu')
    video = animator.run(Image.new('RGB', (48, 48), (90, 60, 200)), 'a boat drifts', 9, 1, (32, 32), steps=2,
                         extra={'max_sequence_length': 16})
    assert video.shape[1:] == (32, 32, 3)
    animator.close()
