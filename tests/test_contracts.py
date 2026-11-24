import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from agentreel.models import Brief,EditorialReview
from agentreel.runtime import Run,verify_log,ROOT
from agentreel.knowledge import official,uploaded_chunks,safe_url
from agentreel.pipeline import approve
from agents.editor import stamp

def test_brief_contract_and_branding():
    assert Brief(topic='Somaiya programmes').brand=='somaiya'
    assert Brief(topic='How rainbows form').brand=='none'
    assert Brief(topic='Rainbows',id='A03').sensitive
    assert Brief(topic='Responsible AI use').sensitive
    for value in (29,61):
        with pytest.raises(ValidationError): Brief(topic='Rainbows',duration_s=value)
    with pytest.raises(ValidationError): Brief(topic='Rainbows',fps=24)
    with pytest.raises(ValidationError): Brief(topic='Rainbows',team_id='../escape')

def test_review_cannot_pass_with_issues():
    with pytest.raises(ValidationError):
        EditorialReview(passed=True,issues=['unsupported'],unsupported_claims=[],must_include_satisfied=True,must_avoid_satisfied=True,contact_checks=[])

def test_hash_chain_detects_mutation_and_intervention_limit(tmp_path):
    run=Run(tmp_path)
    for i in range(5): run.intervention('scriptwriter',f'Change {i}','Human edit')
    assert verify_log(run.logfile)==5
    with pytest.raises(ValueError,match='five'): run.intervention('editor','Sixth','No')
    rows=run.logfile.read_text().splitlines(); row=json.loads(rows[0]); row['change']='tampered'; rows[0]=json.dumps(row)
    run.logfile.write_text('\n'.join(rows),encoding='utf-8')
    with pytest.raises(ValueError,match='integrity'): verify_log(run.logfile)

def test_source_allowlist_and_private_url_rejection():
    assert official('https://kjsce.somaiya.edu/en/')
    assert official('https://www.education.gov.in/')
    assert not official('https://somaiya.edu.attacker.example/')
    with pytest.raises(ValueError): safe_url('http://127.0.0.1/private')
    with pytest.raises(ValueError): safe_url('file:///secret')

def test_document_hash_and_page_metadata(tmp_path):
    p=tmp_path/'reference.md'; p.write_text('An official source paragraph about engineering learning. '*15)
    rows=uploaded_chunks([p]); assert rows and rows[0]['approved_source']
    assert len(rows[0]['content_sha256'])==64 and rows[0]['url']==p.as_uri()

def test_pdf_ingestion_preserves_page(tmp_path):
    from reportlab.pdfgen import canvas
    p=tmp_path/'source.pdf'; c=canvas.Canvas(str(p))
    for page in range(2):
        c.drawString(40,700,'Test source paragraph with enough text for extraction and verified page metadata in our retrieval test.')
        c.showPage()
    c.save(); rows=uploaded_chunks([p])
    assert {r['page'] for r in rows}=={1,2}

def test_srt_rounding():
    assert stamp(59.9996)=='00:01:00,000'
    assert stamp(2.5)=='00:00:02,500'

@pytest.mark.skipif(not (ROOT/'.cache/embeddings/model.onnx').exists(),reason='Local model is downloaded separately')
def test_semantic_paraphrase_and_persistent_faiss():
    from agentreel.embeddings import SemanticStore
    rows=[{'text':'A dog is a domesticated animal often kept as a loyal household pet.','chunk_id':'pet'},
          {'text':'A relational database stores information in tables and indexes.','chunk_id':'database'}]
    first=SemanticStore(rows); scores=first.scores(['a canine companion living with a family'])
    assert scores[0]>scores[1]
    assert first.index.d==384 and first.index.ntotal==2
    again=SemanticStore(rows)
    assert again.fingerprint==first.fingerprint

def test_approval_rejects_unreviewed_or_changed_run(tmp_path):
    (tmp_path/'state.json').write_text(json.dumps({'status':'failed'}))
    with pytest.raises(ValueError,match='not awaiting'): approve(tmp_path,'Reviewer')
    for name in ('brief.yaml','facts.json','script.json','storyboard.json','draft.mp4','qa_report.json'):
        (tmp_path/name).write_text('changed')
    (tmp_path/'state.json').write_text(json.dumps({'status':'awaiting_human_review','review_files':{}}))
    with pytest.raises(ValueError,match='changed'): approve(tmp_path,'Reviewer')
