from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator
import re

class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid')

class Brief(Strict):
    id: str = Field(default='CUSTOM',pattern=r'^[A-Za-z0-9_-]{1,32}$')
    team_id: str = Field(default='T01',pattern=r'^[A-Za-z0-9_-]{1,24}$')
    topic: str = Field(min_length=3,max_length=500)
    audience: str = 'prospective students'
    category: str = 'General'
    format: Literal['video_9x16','video_16x9'] = 'video_9x16'
    duration_s: int = Field(default=40,ge=30,le=60)
    tone: str = 'warm, clear, engaging'
    language: str = 'English'
    subtitle_language: str = 'English'
    must_include: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)
    cta: str = ''
    brand: Literal['auto','somaiya','none'] = 'auto'
    mode: Literal['assignment','general'] = 'assignment'
    fps: Literal[25,30] = 30
    voice: str = ''
    visual_template: Literal['storyboard','rainbow_optics'] = 'storyboard'
    @model_validator(mode='after')
    def resolve_brand(self):
        if self.brand=='auto':
            self.brand='somaiya' if re.search(r'somaiya|kjsce|kjsse',self.topic,re.I) else 'none'
        return self
    @property
    def sensitive(self):
        return bool(re.match(r'^[AC]\d\d$',self.id,re.I) or re.search(r'safety|wellbeing|responsible ai|ragging|mental.health|posh|\bicc\b|suicide|self.harm|drug|cyber|deepfake|academic.integrity|ai.policy|dpdp',self.category+' '+self.topic,re.I))
    @property
    def size(self): return (1080,1920) if self.format=='video_9x16' else (1920,1080)

class Plan(Strict):
    creative_direction: str
    research_queries: list[str] = Field(min_length=3,max_length=8)
    success_criteria: list[str]

class Fact(Strict):
    id: str
    claim: str
    evidence_id: str
    quote: str = Field(min_length=12)

class Facts(Strict):
    facts: list[Fact] = Field(min_length=4,max_length=12)
    gaps: list[str] = Field(default_factory=list)

class Beat(Strict):
    narration: str
    caption: str
    headline: str = Field(max_length=65)
    fact_ids: list[str]

class Script(Strict):
    beats: list[Beat] = Field(min_length=6,max_length=10)
    cta: str = Field(max_length=140)
    must_include_coverage: dict[str,str]

class Pose(Strict):
    time: float = Field(ge=0,le=1)
    x: float = Field(ge=0,le=1)
    y: float = Field(ge=0,le=1)
    scale: float = Field(default=1,ge=.05,le=3)
    angle: float = Field(default=0,ge=-1080,le=1080)
    opacity: float = Field(default=1,ge=0,le=1)

SceneColor = Literal['#4FA5DF','#F4BD50','#5BC58C','#E87470','#EEE8D5','#9D83D2',
                     '#3FC1C9','#F28C38','#E85D9E','#A3D65C','#D94848','#FFFFFF','#5B6B8C','#2E5EAA','#8B5A2B','#FFD93D']
# Generic illustrated building blocks; see agentreel/icons.py.
SHAPES = ('sphere','box','ring','leaf','cloud','person','building','vessel','molecule','wave','disc',
          'gear','bolt','flame','star','heart','shield','lock','screen','document','bulb','clock','globe','coin',
          'flask','atom','dna','drop','tree','rocket','chip','sun','mountain','magnet','crystal','virus','arrow',
          'battery','cell','text')
Shape = Literal[SHAPES]

class SceneObject(Strict):
    id: str = Field(pattern=r'^[a-z][a-z0-9_]{0,31}$')
    subject: str = Field(min_length=3,max_length=120)
    shape: Shape
    color: SceneColor
    size: float = Field(ge=.025,le=.35)
    depth: float = Field(default=.5,ge=0,le=1)
    # Several copies arranged as a group: many coins, people, particles or cells.
    count: int = Field(default=1,ge=1,le=9)
    # Text objects may carry a full equation or short formula; other labels stay terse.
    label: str = Field(default='',max_length=48)
    poses: list[Pose] = Field(min_length=2,max_length=8)
    @model_validator(mode='after')
    def terse_label(self):
        if self.shape!='text' and len(self.label)>24: raise ValueError('Non-text object labels must be at most 24 characters')
        if self.shape=='text' and not self.label.strip(): raise ValueError('Text objects need a label')
        return self

class SceneLink(Strict):
    source: str
    target: str
    meaning: str = Field(min_length=3,max_length=120)
    color: SceneColor
    style: Literal['arrow','flow','line','ray']
    start: float = Field(default=0,ge=0,le=1)
    end: float = Field(default=1,ge=0,le=1)

class ScenePlan(Strict):
    setting: Literal['studio','space','landscape','interior','ocean','city','digital','micro','sky','lab']
    explanation: str = Field(min_length=20,max_length=500)
    # States a visible change, rather than a decorative camera effect.
    visible_change: str = Field(min_length=15,max_length=300)
    objects: list[SceneObject] = Field(min_length=2,max_length=12)
    links: list[SceneLink] = Field(default_factory=list,max_length=16)
    @model_validator(mode='after')
    def executable(self):
        ids=[o.id for o in self.objects]
        if len(ids)!=len(set(ids)): raise ValueError('Scene object IDs must be unique')
        changing=False
        for obj in self.objects:
            times=[p.time for p in obj.poses]
            if times[0]!=0 or times[-1]!=1 or any(b<=a for a,b in zip(times,times[1:])):
                raise ValueError('Object poses must run strictly from time 0 to time 1')
            first=obj.poses[0]
            if obj.shape!='text' and any(abs(p.x-first.x)+abs(p.y-first.y)>.06 or abs(p.scale-first.scale)>.15 or abs(p.angle-first.angle)>20 for p in obj.poses[1:]): changing=True
        for link in self.links:
            if link.source not in ids or link.target not in ids or link.source==link.target:
                raise ValueError('Scene links must reference two existing distinct objects')
            if link.end<=link.start: raise ValueError('Link end must follow start')
            if link.style in ('flow','ray'): changing=True
        if not changing: raise ValueError('Scene needs a visible object action or process, not just text/camera movement')
        return self

class Shot(Strict):
    beat_index: int = Field(ge=0)
    visual: Literal['orbit','network','path','layers','signal','grid']
    visual_description: str
    label: str = Field(max_length=32)
    accent: Literal['red','teal','gold','blue']
    stock_query: str = Field(default='', max_length=100)
    # Broader fallbacks tried in order when the primary query has no relevant clip.
    stock_alternates: list[str] = Field(default_factory=list, max_length=3)
    # footage: real clip when a relevant one exists; animation: the scene graph explains it;
    # illustration: a detailed local AI still (image_prompt) with depth parallax.
    visual_mode: Literal['footage','animation','illustration'] = 'footage'
    image_prompt: str = Field(default='', max_length=400)
    # Up to two more angles or moments of the same beat (wide, close-up, reaction); the
    # renderer cuts between them so an illustrated beat plays like edited footage.
    extra_image_prompts: list[str] = Field(default_factory=list, max_length=2)
    # What moves in an illustration once a local image-to-video model animates it.
    motion_prompt: str = Field(default='', max_length=300)
    scene: ScenePlan | None = None
    transition: Literal['cut','dissolve'] = 'dissolve'
    demonstration: Literal['auto','optics','gears','signal_flow','pathway','assembly','network'] = 'auto'
    camera: Literal['push_in','track_left','track_right','orbit'] = 'push_in'
    @model_validator(mode='before')
    @classmethod
    def hoist_shot_fields(cls,data):
        # A scene left unclosed in model JSON swallows the shot's trailing fields.
        if isinstance(data,dict) and isinstance(data.get('scene'),dict):
            scene=dict(data['scene']); data=dict(data)
            for key in [k for k in scene if k in cls.model_fields and k not in ScenePlan.model_fields]:
                data.setdefault(key,scene.pop(key))
            data['scene']=scene
        return data

Theme = Literal['midnight','ocean','sunset','forest','neon','paper','ember','arctic']

class Storyboard(Strict):
    # One colour world for the whole film, chosen to suit the topic and tone.
    theme: Theme = 'midnight'
    shots: list[Shot] = Field(min_length=6,max_length=10)

class EditorialReview(Strict):
    passed: bool
    issues: list[str]
    unsupported_claims: list[str]
    must_include_satisfied: bool
    must_avoid_satisfied: bool
    contact_checks: list[str]
    @model_validator(mode='after')
    def consistent(self):
        if self.passed and (self.issues or self.unsupported_claims or not self.must_include_satisfied or not self.must_avoid_satisfied):
            raise ValueError('Passing review cannot contain unresolved issues')
        return self
