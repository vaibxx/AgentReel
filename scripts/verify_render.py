"""Developer integration check: rebuild an existing approved editorial draft locally.

Does not re-research content or approve sensitive outputs. Records the rerender.
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from dotenv import load_dotenv
from agentreel.runtime import Run,ROOT,sha
from agentreel.models import Brief,Script
from agentreel.pipeline import export
from agentreel import qa
from agents import editor

def main():
    load_dotenv(ROOT/'.env'); ctx=Run(sys.argv[1]); state=json.loads((ctx.path/'state.json').read_text())
    brief=Brief.model_validate(yaml.safe_load((ctx.path/'brief.yaml').read_text()))
    if brief.sensitive: raise ValueError('Sensitive drafts require the normal generation/review workflow.')
    reviews=sorted(ctx.path.glob('editorial_review_*.json'))
    if not reviews or not json.loads(reviews[-1].read_text())['passed']: raise ValueError('No passing editorial review.')
    records=[json.loads(line) for line in ctx.logfile.read_text(encoding='utf-8').splitlines()]
    for name in ('script.json','storyboard.json','facts.json'):
        previous=[r for r in records if r.get('current_path')==name][-1]
        if sha(ctx.path/name)!=previous['output_sha256']: raise ValueError('Editorial input changed: '+name)
    script=Script.model_validate_json((ctx.path/'script.json').read_text(encoding='utf-8'))
    timings=json.loads((ctx.path/'timings.json').read_text())['segments']; assets=ctx.path/'assets'
    frames=[(assets/f'background_{i:02}.png',assets/f'overlay_{i:02}.png') for i in range(len(timings))]+[(assets/'end_background.png',assets/'end_overlay.png')]
    music=assets/'music.wav'; music=music if music.exists() else None
    ctx.emit('editor','development_render_verification',inputs={'reason':'Correct CFR timestamp handling'})
    draft=editor.run(brief,script,ctx,timings,assets/'end_silence.wav',frames,music,[])
    result=qa.run(ctx,brief,draft)
    if not result['passed']: raise ValueError('QA failed: '+str(result['checks']))
    print(export(ctx,brief,state),flush=True)
if __name__=='__main__': main()
