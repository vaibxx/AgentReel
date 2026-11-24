import re
from agentreel.models import Facts
from agentreel.knowledge import Knowledge,web_research,norm

FOLD=str.maketrans({'‘':"'",'’':"'",'“':'"','”':'"','–':'-','—':'-',' ':' '})

def verbatim(quote,text):
    """Quote must appear in the evidence; an excerpt joined with '...' passes when every
    piece appears in order. Typographic quotes and dashes are folded, nothing else."""
    text=norm(text.translate(FOLD)).lower(); at=0
    pieces=[norm(p) for p in re.split(r'\.{3}|…',quote.translate(FOLD).lower())]
    pieces=[p.strip(' .,;:') for p in pieces if len(p.strip(' .,;:'))>=8]
    if not pieces: return False
    for piece in pieces:
        found=text.find(piece,at)
        if found<0: return False
        at=found+len(piece)
    return True

def run(brief,plan,provider,ctx,documents):
    knowledge=Knowledge(ctx,brief.brand,documents)
    evidence=knowledge.retrieve(plan.research_queries)
    if not evidence: evidence=web_research(ctx,plan.research_queries,brief.mode=='assignment' and brief.brand=='somaiya')
    evidence=[{**r,'evidence_id':f'E{i+1:03}'} for i,r in enumerate(evidence)]
    ctx.artifact('evidence.json',evidence,'researcher')
    feedback=''
    for attempt in range(3):
        result=provider.generate('researcher',Facts,{'brief':brief.model_dump(),'evidence':evidence,'require_provided_policy_for_contacts':brief.sensitive},feedback)
        mapping={r['evidence_id']:r for r in evidence}; errors={}; seen=set(); unique=[]
        for fact in result.facts:
            # A repeated ID keeps its first fact; later copies are dropped.
            if fact.id in seen: continue
            seen.add(fact.id); unique.append(fact)
        result.facts=unique
        for fact in result.facts:
            e=mapping.get(fact.evidence_id); problems=[]
            if not e or not verbatim(fact.quote,e['text']): problems.append('quote is not verbatim evidence')
            if set(re.findall(r'\d+(?:[.,]\d+)*',fact.claim))-set(re.findall(r'\d+(?:[.,]\d+)*',fact.quote)): problems.append('numerical claim not present in its quote')
            if brief.sensitive and re.search(r'@|\b\d{5,}\b|helpline|\bcontact\b',fact.claim,re.I) and e and not e.get('approved_source'):
                problems.append('safety contact must come from a provided official policy/fact sheet')
            if problems: errors[fact.id]=problems
        kept=[f for f in result.facts if f.id not in errors]
        # Unverifiable facts are removed, never repaired by guessing; enough verified
        # facts still make a grounded script. The removals stay on record.
        if not errors or len(kept)>=4:
            data=result.model_dump(); data['facts']=[f.model_dump() for f in kept]
            for f in data['facts']:
                e=mapping[f['evidence_id']]; f['source']={k:e.get(k) for k in ('source_id','chunk_id','url','page','section','retrieved_at','content_sha256','approved_source')}
            if errors:
                data['rejected_facts']=[{'id':f.id,'claim':f.claim,'reasons':errors[f.id]} for f in result.facts if f.id in errors]
                ctx.emit('researcher','drop_unverified_facts',inputs={'dropped':sorted(errors),'kept':len(kept)})
            ctx.artifact('facts.json',data,'researcher'); return data
        feedback='\n'.join(f'{k}: {"; ".join(v)}.' for k,v in errors.items())+'\nQuotes must be exact substrings of one evidence passage (you may join exact excerpts with "...").'
    raise ValueError('Research evidence checks failed: '+feedback)
