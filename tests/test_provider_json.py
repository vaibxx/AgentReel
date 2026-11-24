import json
import pytest
from agentreel.provider import lenient_json
from agentreel.models import Shot

def test_unclosed_object_before_sibling_and_parent_is_repaired():
    broken='{"shots": [{"a": {"x": 1, "t": "cut"}, {"a": {"x": 2, "t": "cut"}]}'
    assert json.loads(lenient_json(broken))=={'shots':[{'a':{'x':1,'t':'cut'}},{'a':{'x':2,'t':'cut'}}]}

def test_brackets_inside_strings_are_ignored_and_tail_is_balanced():
    assert json.loads(lenient_json('{"a": [1, {"b": "x]}"'))=={'a':[1,{'b':'x]}'}]}

def test_unrepairable_json_still_fails():
    with pytest.raises(json.JSONDecodeError): lenient_json('{"a": nope}')

def test_shot_fields_swallowed_by_scene_are_hoisted():
    shot=Shot.model_validate({'beat_index':0,'visual':'orbit','visual_description':'d','label':'l','accent':'blue',
                              'scene':{'setting':'studio','explanation':'x'*20,'visible_change':'y'*15,'objects':[
                                  {'id':'a','subject':'ball','shape':'sphere','color':'#4FA5DF','size':.1,'poses':[{'time':0,'x':.2,'y':.5},{'time':1,'x':.6,'y':.5}]},
                                  {'id':'b','subject':'box','shape':'box','color':'#4FA5DF','size':.1,'poses':[{'time':0,'x':.8,'y':.5},{'time':1,'x':.8,'y':.5}]}],
                                  'transition':'cut','camera':'orbit','visual_mode':'animation'}})
    assert (shot.transition,shot.camera,shot.visual_mode)==('cut','orbit','animation')

def test_trailing_garbage_after_complete_document_is_dropped():
    assert json.loads(lenient_json('{"shots": [{"a": 1}]}\n]}'))=={'shots':[{'a':1}]}
