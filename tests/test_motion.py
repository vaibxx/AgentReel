import numpy as np
from agentreel.models import Brief,Beat,Shot,Script
from agentreel.motion import Scene,caption_cues,media_for_beat

def beat(text='Light enters a water droplet and changes direction.'):
    return Beat(narration=text,caption=text,headline='Inside the droplet',fact_ids=[])

def test_dispersion_phase_uses_storyboard_intent():
    b=beat('This bending separates white light into a vibrant color spectrum.')
    shot=Shot(beat_index=0,visual='path',visual_description='Light separates inside the droplet',label='Dispersion',accent='red',demonstration='optics')
    assert Scene(Brief(topic='rainbows'),b,shot,0,(360,640)).optic_phase=='spectrum'

def test_phrase_captions_preserve_words_and_timeline():
    b=beat(); script=Script(beats=[b]*6,cta='Explore',must_include_coverage={})
    times=[{'index':i,'start':i*4.,'end':i*4.+4,'spoken_end':i*4.+3.8} for i in range(6)]
    cues=caption_cues(script,times)
    assert len(cues)>6 and cues[0]['start']==0 and cues[-1]['end']==24
    for i in range(6): assert ' '.join(c['text'] for c in cues if c['beat_index']==i)==b.caption
    assert all(a['end']<=b['start']+1e-8 for a,b in zip(cues,cues[1:]))

def test_same_language_captions_do_not_omit_spoken_words():
    from agentreel.motion import render_script
    b=beat(); b.caption='Short summary'
    script=Script(beats=[b]*6,cta='Explore',must_include_coverage={})
    corrected=render_script(Brief(topic='Optics'),script)
    assert corrected.beats[0].caption==b.narration
    assert script.beats[0].caption=='Short summary'
    translated=render_script(Brief(topic='Optics',subtitle_language='Hindi'),script)
    assert translated.beats[0].caption=='Short summary'

def test_media_is_matched_and_not_repeated_unrelated():
    media=[{'path':'mechanical.mp4','beat_index':'2'},{'path':'chip.mp4','description':'computer circuit engineering'}]
    assert media_for_beat(media,2,beat())['path']=='mechanical.mp4'
    assert media_for_beat(media,0,beat('Computer circuit engineering demonstration'))['path']=='chip.mp4'
    assert media_for_beat(media,5,beat('Rain falls from a cloud')) is None

def test_actual_scene_geometry_moves_at_fixed_global_camera():
    b=beat('Mechanical gears transmit movement.')
    shot=Shot(beat_index=0,visual='network',visual_description='Meshing gears',label='Mechanics',accent='teal',demonstration='gears')
    scene=Scene(Brief(topic='Mechanical engineering'),b,shot,0,(360,640))
    # Same global time fixes atmospheric motion; rotating mechanism must still change.
    first=np.asarray(scene.frame(1,2)); second=np.asarray(scene.frame(2,2))
    assert np.mean(np.abs(first.astype(float)-second))>.5
    assert scene.kind=='gears'

def test_local_video_decodes_moving_frames(tmp_path):
    import shutil,subprocess,pytest
    from agentreel.motion import Footage
    ff=shutil.which('ffmpeg')
    if not ff: pytest.skip('FFmpeg integration dependency not installed')
    clip=tmp_path/'fixture.mp4'
    subprocess.run([ff,'-y','-v','error','-f','lavfi','-i','testsrc2=size=160x90:rate=30','-t','1','-c:v','libx264','-pix_fmt','yuv420p',str(clip)],check=True)
    reader=Footage({'path':str(clip)},(160,90),30)
    try:
        first=np.asarray(reader.frame(0))
        for i in range(10): second=np.asarray(reader.frame((i+1)/30))
        assert np.mean(np.abs(first.astype(float)-second))>1
    finally: reader.close()

def test_commons_import_rejects_unapproved_licence(monkeypatch,tmp_path):
    import pytest
    from scripts import import_commons_video as importer
    class Reply:
        def raise_for_status(self): pass
        def json(self): return {'query':{'pages':{'1':{'imageinfo':[{'extmetadata':{'License':{'value':'cc-by-sa-4.0'},'LicenseShortName':{'value':'CC BY-SA'}}}]}}}}
    monkeypatch.setattr(importer.requests,'get',lambda *a,**kw:Reply())
    monkeypatch.setattr(importer.sys,'argv',['import','--title','File:test.webm','--beat-index','0','--output-dir',str(tmp_path)])
    with pytest.raises(ValueError,match='CC0'): importer.main()
    assert not list(tmp_path.iterdir())

def test_generated_clip_stretches_and_holds_instead_of_looping(tmp_path):
    import shutil,subprocess,pytest
    from agentreel.motion import Footage
    ff=shutil.which('ffmpeg')
    if not ff: pytest.skip('FFmpeg integration dependency not installed')
    clip=tmp_path/'video_02.mp4'
    subprocess.run([ff,'-y','-v','error','-f','lavfi','-i','testsrc2=size=160x96:rate=24','-t','1','-c:v','libx264','-pix_fmt','yuv420p',str(clip)],check=True)
    # A 1 s clip for a 3 s beat: slowed to 1.6 s, then the last frame holds.
    reader=Footage({'path':str(clip),'clip_s':1.0,'ai_generated':True,'video':True,'beat_index':0},(160,96),30,duration=3.0)
    try:
        frames=[np.asarray(reader.frame(i/30)).astype(float) for i in range(90)]
    finally: reader.close()
    assert np.mean(np.abs(frames[0]-frames[30]))>1          # still moving after 1 s (slowed)
    assert np.mean(np.abs(frames[70]-frames[89]))<.5        # held at the end
    assert np.mean(np.abs(frames[0]-frames[89]))>1          # never jumped back to the first frame
