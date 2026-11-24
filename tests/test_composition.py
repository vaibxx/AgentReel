import numpy as np
import pytest
from pydantic import ValidationError
from agentreel.models import ScenePlan
from agentreel.composition import Composition,pose_at

def plan(flow=False):
    def obj(name,x,end,shape='sphere'):
        return dict(id=name,subject=name+' demonstration object',shape=shape,color='#4FA5DF',size=.12,
                    poses=[dict(time=0,x=x,y=.5),dict(time=1,x=end,y=.5)])
    return dict(setting='studio',explanation='Objects approach each other to illustrate an interaction.',
                visible_change='The distance between objects decreases.',
                objects=[obj('left',.2,.4),obj('right',.8,.6)],
                links=[dict(source='left',target='right',meaning='Material transfer',color='#F4BD50',style='flow')] if flow else [])

def test_object_action_changes_geometry_without_camera():
    graph=ScenePlan.model_validate(plan())
    renderer=Composition(graph,(640,360),'orbit',4)
    a=renderer.positions(0);b=renderer.positions(1)
    assert b['right'][2]-b['left'][2]<a['right'][2]-a['left'][2]
    assert np.abs(np.asarray(renderer.frame(0)).astype(float)-np.asarray(renderer.frame(4))).mean()>1

def test_static_shapes_do_not_pass_as_an_explanation():
    graph=plan()
    for obj in graph['objects']: obj['poses'][1]['x']=obj['poses'][0]['x']
    with pytest.raises(ValidationError,match='visible object action'):ScenePlan.model_validate(graph)

def test_camera_and_text_cannot_satisfy_action_requirement():
    graph=plan()
    for obj in graph['objects']:obj['shape']='text'; obj['label']='E = mc²'
    with pytest.raises(ValidationError,match='visible object action'):ScenePlan.model_validate(graph)

def test_broken_links_and_unsorted_timelines_rejected():
    graph=plan(True);graph['links'][0]['target']='missing'
    with pytest.raises(ValidationError,match='reference'):ScenePlan.model_validate(graph)
    graph=plan();graph['objects'][0]['poses'][0]['time']=.5
    with pytest.raises(ValidationError,match='time 0'):ScenePlan.model_validate(graph)

def test_directed_flow_changes_frames_with_fixed_objects():
    graph=plan(True)
    for obj in graph['objects']:obj['poses'][1]['x']=obj['poses'][0]['x']
    renderer=Composition(ScenePlan.model_validate(graph),(640,360),'orbit',4)
    assert not np.array_equal(renderer.frame(.1),renderer.frame(.8))

def test_timing_uses_fraction_of_spoken_beat():
    graph=ScenePlan.model_validate(plan())
    assert pose_at(graph.objects[0],.5)['x']==pytest.approx(.3)
    assert pose_at(graph.objects[0],2)['x']==pytest.approx(.4)

def test_narration_leaves_time_to_observe_the_action():
    from agentreel.speech import pacing
    tempo,gap,_=pacing([2.9]*6,26)
    assert tempo==.9 and .8<gap<1.5
    assert sum([2.9]*6)/tempo+gap*6==pytest.approx(26)

from agentreel.models import SHAPES

@pytest.mark.parametrize('shape',SHAPES)
def test_reusable_geometry_renders_without_topic_branch(shape):
    graph=plan();graph['objects'][0].update(shape=shape,label='Subject')
    renderer=Composition(ScenePlan.model_validate(graph),(640,360),'push_in',4)
    assert renderer.frame(2).size==(640,360)


def test_ray_counts_as_process_and_reveals_over_time():
    graph=plan()
    for obj in graph['objects']: obj['poses'][1]['x']=obj['poses'][0]['x']
    graph['links']=[dict(source='left',target='right',meaning='Light travels',color='#F4BD50',style='ray')]
    renderer=Composition(ScenePlan.model_validate(graph),(640,360),'push_in',4)
    assert np.abs(np.asarray(renderer.frame(.2)).astype(float)-np.asarray(renderer.frame(3)).astype(float)).mean()>.5

def test_new_shapes_and_equation_text_render():
    graph=plan()
    for obj,shape in zip(graph['objects'],('molecule','disc')): obj['shape']=shape
    graph['objects'].append(dict(id='eq',subject='equation',shape='text',color='#F4BD50',size=.1,label='6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂',
                                 poses=[dict(time=0,x=.5,y=.05),dict(time=1,x=.5,y=.05)]))
    graph['objects'].append(dict(id='w',subject='signal',shape='wave',color='#4FA5DF',size=.08,poses=[dict(time=0,x=.5,y=.8),dict(time=1,x=.5,y=.8)]))
    frame=Composition(ScenePlan.model_validate(graph),(1080,1920),'push_in',4).frame(2)
    assert frame.size==(1080,1920)

def test_non_text_labels_stay_short():
    graph=plan(); graph['objects'][0]['label']='a label that is far too long for an object'
    with pytest.raises(ValidationError,match='at most 24'): ScenePlan.model_validate(graph)

def test_missing_glyphs_never_render_as_boxes(monkeypatch):
    from agentreel import composition
    from agentreel.visuals import font
    monkeypatch.setattr(composition,'SYMBOL_FONTS',[])
    assert ''.join(s for s,f in composition.runs('A → B',font(40)))=='A -> B'


@pytest.mark.parametrize('theme',['midnight','ocean','sunset','forest','neon','paper','ember','arctic'])
@pytest.mark.parametrize('setting',['studio','space','landscape','interior','ocean','city','digital','micro','sky','lab'])
def test_every_theme_and_setting_renders(theme,setting):
    graph=plan(); graph['setting']=setting
    frame=Composition(ScenePlan.model_validate(graph),(180,320),'push_in',4,theme).frame(1)
    assert frame.size==(180,320)

def test_themes_change_the_look():
    graph=ScenePlan.model_validate(plan())
    dark=np.asarray(Composition(graph,(180,320),'push_in',4,'midnight').frame(1)).mean()
    light=np.asarray(Composition(graph,(180,320),'push_in',4,'paper').frame(1)).mean()
    assert light>dark+80

def test_count_draws_a_group():
    single=plan(); group=plan(); group['objects'][0]['count']=6
    a=np.asarray(Composition(ScenePlan.model_validate(single),(640,360),'push_in',4).frame(1)).astype(float)
    b=np.asarray(Composition(ScenePlan.model_validate(group),(640,360),'push_in',4).frame(1)).astype(float)
    assert np.abs(a-b).mean()>1
