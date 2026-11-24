import re
from agentreel.models import Script
from agentreel.speech import speak,check_length,NarrationLengthError

CAPTION_LIMIT=220  # characters per beat; on screen each beat is split into short phrase cues
WORDS_PER_BEAT=20  # above this, more beats read better than longer ones

def budget(result,target):
    """Per-beat word budgets: models follow these far better than a total count. When
    beats would get too long, ask for more beats (the schema allows up to 10)."""
    beats=len(result.beats)
    if target/beats>WORDS_PER_BEAT and beats<10:
        more=min(10,-(-target//WORDS_PER_BEAT))
        return (f'Use {more} beats instead of {beats} (more, shorter beats keep captions readable and give more shots), '
                f'about {round(target/more)} words each, about {target} words in total.')
    each=max(4,round(target/beats))
    return 'Per-beat budget: '+'; '.join(f'beat {i+1} has {len(b.narration.split())} words -> use about {each}' for i,b in enumerate(result.beats))+'.'
def run(brief,facts,provider,ctx,feedback='',target=None):
    target=target or round((brief.duration_s-4)*2.25); base=feedback
    for attempt in range(3):
        result=provider.generate('scriptwriter',Script,{'brief':brief.model_dump(),'facts':facts,'target_word_count':target,'minimum_words':round(target*.88),'maximum_words':round(target*1.08)},feedback)
        if brief.language.strip().casefold()==brief.subtitle_language.strip().casefold():
            for beat in result.beats: beat.caption=beat.narration
        ids={f['id'] for f in facts['facts']}; errors=[]
        words=sum(len(b.narration.split()) for b in result.beats)
        # Only a coarse guard; the measured narration audio below is the real length check.
        if not target*.6<=words<=target*1.5: errors.append(f'Narration has {words} words; aim for {target} words in total. '+budget(result,target))
        if any(not set(b.fact_ids)<=ids for b in result.beats): errors.append('Unknown fact IDs.')
        if set(brief.must_include)-set(result.must_include_coverage): errors.append('Missing must_include coverage entries.')
        long=[i+1 for i,b in enumerate(result.beats) if len(b.caption)>CAPTION_LIMIT]
        if long: errors.append(f'Beats {long} have captions over {CAPTION_LIMIT} characters. Split them into more beats rather than shortening the content; same-language captions must stay verbatim.')
        urls=re.findall(r'(?:https?://|www\.)\S+|\b\S+\.(?:com|org|edu|in|net|io)\S*',result.cta,re.I)
        if urls and brief.brand=='none': errors.append('Unbranded videos must not send viewers to a third-party website in the CTA; end with a spoken invitation instead.')
        if any(len(u)>40 for u in urls): errors.append('CTA URL is too long for the end card; use the short official domain only.')
        if not errors:
            # Measure the draft with the real voice; speaking rate depends on vocabulary,
            # so the word target is corrected from this measurement, not a fixed guess.
            try: check_length(brief,result,speak(ctx,brief,result)[1])
            except NarrationLengthError as exc:
                target=exc.target_words; errors.append(str(exc)+' '+budget(result,target))
        if not errors:
            ctx.artifact('script.json',result.model_dump(),'scriptwriter')
            (ctx.path/'script.md').write_text('# Script\n\n'+'\n\n'.join(f'## Beat {i+1}\n{b.narration}\n\nEvidence: {", ".join(b.fact_ids)}' for i,b in enumerate(result.beats))+'\n\nCTA: '+result.cta,encoding='utf-8')
            return result
        # Replace, not accumulate: stale word targets in old feedback contradict new ones.
        feedback='\n'.join([base,*errors]).strip()
    raise ValueError('Script validation failed: '+feedback)
