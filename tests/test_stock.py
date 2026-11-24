import json
import os
import time
from types import SimpleNamespace
import pytest
import requests
from agentreel import stock
from agentreel.models import Brief

def test_search_cache_and_error_redaction(tmp_path, monkeypatch):
    monkeypatch.setenv('PIXABAY_API_KEY', 'secret-test-value')
    calls=[]
    def get(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=200, json=lambda: {'hits':[{'id':1}]})
    monkeypatch.setattr(stock.requests, 'get', get)
    assert stock.search('rainbow', tmp_path)==[{'id':1}]
    assert stock.search('rainbow', tmp_path)==[{'id':1}]
    assert len(calls)==1
    path=next(tmp_path.glob('*.json'))
    assert 'secret-test-value' not in path.read_text()
    os.utime(path,(time.time()-86401,)*2)
    def fail(*args, **kwargs): raise requests.ConnectionError('secret-test-value')
    monkeypatch.setattr(stock.requests,'get',fail)
    with pytest.raises(RuntimeError) as error: stock.search('rainbow',tmp_path)
    assert 'secret-test-value' not in str(error.value)

def test_no_unrelated_or_duplicate_candidates():
    hits=[dict(id=1,tags='city, bridge',videos={'medium':dict(url='x',size=100)}),
          dict(id=2,tags='rainbow, landscape',videos={'medium':dict(url='x',size=100)})]
    assert [h['id'] for h,v in stock.candidates(hits,'rainbow',set())]==[2]
    assert stock.candidates(hits,'rainbow',{2})==[]

def test_assignment_and_somaiya_never_request_stock(monkeypatch):
    def forbidden(*args): raise AssertionError('Network must not run')
    monkeypatch.setattr(stock,'search',forbidden)
    for brief in (Brief(topic='Rainbow'),Brief(topic='Somaiya',mode='general')):
        assert stock.collect(brief,None,None,[])==[]
        with pytest.raises(ValueError): stock.collect(brief,None,None,[],source='pixabay')

def test_download_rejects_untrusted_host(tmp_path):
    with pytest.raises(ValueError): stock.download('https://example.com/clip.mp4',tmp_path/'clip.mp4')

def test_manual_media_prevents_automatic_replacement(monkeypatch,tmp_path):
    monkeypatch.setenv('PIXABAY_API_KEY','test')
    monkeypatch.setattr(stock,'search',lambda q: pytest.fail('Manual media should win'))
    ctx=SimpleNamespace(path=tmp_path,artifact=lambda *args:None)
    board=SimpleNamespace(shots=[SimpleNamespace(stock_query='rainbow')])
    assert stock.collect(Brief(topic='rainbow',mode='general'),board,ctx,[{'path':'custom.mp4'}])==[]


def test_incidental_rainbow_tag_does_not_match_balloon():
    hit=dict(id=3,tags='hot air balloon, flying, valley, sky, rainbow, balloon in sky',videos={'medium':dict(url='x',size=100)})
    assert stock.candidates([hit],'rainbow in sky',set())==[]


def _hit(id, tags, w=1920, h=1080, duration=20):
    return dict(id=id, tags=tags, duration=duration, pageURL=f'https://pixabay.com/videos/{id}/',
                videos={'large':dict(url=f'https://cdn.pixabay.com/{id}.mp4',size=1000,width=w,height=h)})

def test_partial_and_plural_queries_match_subject():
    hits=[_hit(1,'rain, water, wet, drops, raindrop'),_hit(2,'city, traffic, night')]
    assert [h['id'] for h,v in stock.candidates(hits,'water drop close up',set())]==[1]

def test_blocked_tags_and_must_avoid_are_never_chosen():
    hits=[_hit(1,'skin, eroticism, waterdrop, water'),_hit(2,'water, splash, drops'),_hit(3,'water, crowd, fear')]
    assert [h['id'] for h,v in stock.candidates(hits,'water',set(),blocked={'fear'})]==[2]

def test_orientation_and_duration_rank_candidates():
    hits=[_hit(1,'forest, trees',duration=3),_hit(2,'forest, trees',w=1080,h=1920,duration=20)]
    assert [h['id'] for h,v in stock.candidates(hits,'forest',set(),vertical=True,need_s=6)][0]==2

def test_each_shot_falls_back_through_alternates(monkeypatch,tmp_path):
    monkeypatch.setenv('PIXABAY_API_KEY','test')
    results={'quantum foam':[], 'particle physics':[], 'galaxy':[_hit(7,'galaxy, stars, space')]}
    monkeypatch.setattr(stock,'search',lambda q: results.get(q,[]))
    monkeypatch.setattr(stock,'fetch',lambda hit,video,target: target.write_bytes(b'clip'))
    (tmp_path/'assets').mkdir()
    events=[]
    ctx=SimpleNamespace(path=tmp_path,artifact=lambda *args:None,emit=lambda *a,**k:events.append(a[1]))
    shot=SimpleNamespace(beat_index=0,stock_query='quantum foam',stock_alternates=['particle physics','galaxy'])
    records=stock.collect(Brief(topic='space',mode='general'),SimpleNamespace(shots=[shot]),ctx,[])
    assert [r['query'] for r in records]==['galaxy'] and records[0]['beat_index']==0

def test_homonym_with_low_coverage_is_rejected():
    hits=[_hit(1,'cell phone, internet, technology'),_hit(2,'cell, membrane, biology, microscope')]
    assert [h['id'] for h,v in stock.candidates(hits,'cell membrane layers',set())]==[2]

def test_animation_shots_never_search(monkeypatch,tmp_path):
    monkeypatch.setenv('PIXABAY_API_KEY','test')
    monkeypatch.setattr(stock,'search',lambda q: pytest.fail('Animation shots must not search'))
    ctx=SimpleNamespace(path=tmp_path,artifact=lambda *args:None,emit=lambda *a,**k:None)
    shot=SimpleNamespace(beat_index=0,stock_query='lab',stock_alternates=[],visual_mode='animation')
    assert stock.collect(Brief(topic='atp',mode='general'),SimpleNamespace(shots=[shot]),ctx,[])==[]

def test_style_adjective_cannot_carry_a_match():
    hits=[_hit(1,'photography, laboratory, studio, woman, vintage'),_hit(2,'laboratory, equipment, glassware, chemistry')]
    assert [h['id'] for h,v in stock.candidates(hits,'vintage laboratory equipment',set())]==[2]
