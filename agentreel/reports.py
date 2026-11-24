import csv,json,shutil
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape
from PIL import Image,ImageDraw
from .visuals import font
from .runtime import ROOT

def graph(path):
    im=Image.new('RGB',(1600,720),'#0c1522'); d=ImageDraw.Draw(im)
    d.text((55,35),'AgentReel · evidence to finished video',font=font(38),fill='white')
    labels=['Planner','Researcher / RAG','Scriptwriter','Storyboarder','Asset Generator','Editor','Critic / QA']
    for i,label in enumerate(labels):
        x=50+(i%4)*390; y=140+(i//4)*230
        d.rounded_rectangle((x,y,x+340,y+150),radius=22,fill='#162c42',outline='#319c9a',width=3)
        d.text((x+20,y+30),label,font=font(24),fill='white')
        d.text((x+20,y+80),['Brief -> plan','Sources -> cited facts','Facts -> timed script','Script -> visual shots','Voice / art / captions','1080p MP4 + SRT','Verify / repair / export'][i],font=font(19),fill='#b7cbdc')
        if i%4<3 and i<6:
            d.line((x+345,y+75,x+380,y+75),fill='#f0b866',width=4)
            d.polygon([(x+382,y+75),(x+372,y+68),(x+372,y+82)],fill='#f0b866')
    d.line([(1390,295),(1390,330),(220,330),(220,365)],fill='#f0b866',width=4)
    d.polygon([(220,367),(213,357),(227,357)],fill='#f0b866')
    d.text((55,625),'Critic failure -> responsible stage (max 3 revisions)   |   A/C safety topics -> human review before export',font=font(22),fill='#f0b866')
    path.parent.mkdir(exist_ok=True,parents=True); im.save(path)

def report(ctx,brief,status):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image as PDFImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    folder=ctx.path/'docs'; folder.mkdir(exist_ok=True)
    graph(folder/'agent_graph.png')
    styles=getSampleStyleSheet(); styles['BodyText'].fontSize=9; styles['BodyText'].leading=12
    doc=SimpleDocTemplate(str(folder/'process_report.pdf'),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=35,bottomMargin=35)
    story=[]
    def p(text,style='BodyText'): story.append(Paragraph(escape(str(text)),styles[style])); story.append(Spacer(1,7))
    p('AgentReel production process report','Title')
    p(f'Team {brief.team_id} | Brief {brief.id} | Run {ctx.path.name}')
    p('Brief and creative strategy','Heading2'); p(brief.topic); p(f'Audience: {brief.audience}. Tone: {brief.tone}. Requested length: {brief.duration_s}s. Format: {brief.format}. Language: {brief.language}.')
    plan=json.loads((ctx.path/'plan.json').read_text(encoding='utf-8')) if (ctx.path/'plan.json').exists() else {}
    p(plan.get('creative_direction','See plan.json.'))
    p('Workflow and evidence','Heading2')
    p('Seven explicit roles exchange validated JSON. Research uses source-linked evidence and exact supporting quotations. Somaiya requests retrieve the local vector collection; other topics use web research or supplied documents. Critic failures feed back into a bounded revision loop.')
    story.append(PDFImage(str(folder/'agent_graph.png'),width=510,height=230))
    p('Output and review','Heading2'); p(f'Status: {status}. Autonomy target: L3 with mandatory review for sensitive topics, L4 for eligible non-sensitive runs. Actual interventions and service failures are retained in the logs.')
    story.append(PageBreak()); p('Time cost and verification','Title')
    totals=defaultdict(lambda:[0.,0,0,0.])
    for row in csv.DictReader(ctx.costfile.open(encoding='utf-8')):
        r=totals[row['stage']]; r[0]+=float(row['latency_s']); r[1]+=int(row['input_tokens']); r[2]+=int(row['output_tokens']); r[3]+=float(row['cost_inr'])
    data=[['Stage','Seconds','Input tokens','Output tokens','INR']]+[[name,f'{v[0]:.1f}',str(v[1]),str(v[2]),f'{v[3]:.2f}'] for name,v in sorted(totals.items())]
    table=Table(data,colWidths=[155,70,90,90,65]); table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#152f46')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke,colors.white])]))
    story.append(table); story.append(Spacer(1,12))
    p('Critic loop and interventions','Heading2')
    reviews=[json.loads(pth.read_text(encoding='utf-8')) for pth in ctx.path.glob('editorial_review_*.json')]
    p(f'{len(reviews)} editorial review passes; {sum(not x["passed"] for x in reviews)} rejected drafts. See the review JSON files for reasons and run_log.jsonl for repair calls. Human changes are in interventions.md; safety approval is a separate signed-off manifest.')
    qa=json.loads((ctx.path/'qa_report.json').read_text(encoding='utf-8')) if (ctx.path/'qa_report.json').exists() else {}
    p('Technical checks: '+(', '.join(k for k,v in qa.get('checks',{}).items() if v) or 'Not yet completed.'))
    p('Limitations ethics and improvement','Heading2')
    p('Free API access depends on the account and quota; local logs cannot verify provider billing. No paid fallback is configured. Neural voices are stock model voices, not imitations of real people. Source licensing and supplied-media consent are recorded in the asset manifest. Captions align to measured spoken beats. Historical figures and safety contacts need the documented source and review checks. Improve with approved campus footage and faculty-reviewed retrieval benchmarks; automated factual review is not infallible.')
    p('Reproducibility','Heading2'); p(f'python run.py --brief "{ctx.path / "brief.yaml"}" --seed {ctx.seed}. Model sampling and external websites can change; cached evidence, prompts, model names and file hashes are preserved.')
    doc.build(story)
    ctx.emit('reporter','generate_process_report',output_path='docs/process_report.pdf')
