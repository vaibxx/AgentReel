"""Bounded Pixabay video retrieval. Credentials never enter logs or cache keys.

Every storyboard shot carries a primary footage query plus broader alternates written by
the storyboarder for this topic. Each shot tries its chain in order and keeps the best
ranked relevant clip; shots without a relevant clip fall back to their animated scene.
No topic-specific code or query lists exist here.
"""
import hashlib
import json
import math
import os
import re
import shutil
import time
from urllib.parse import urlparse
import requests
from agentreel.runtime import ROOT, sha

LICENSE = 'https://pixabay.com/service/license-summary/'
MAX_BYTES = 80 * 1024 * 1024
# Prefer full-HD sources when they stay reasonably small; otherwise the 720p rendition.
LARGE_LIMIT = 45 * 1024 * 1024
MAX_SEARCHES = 40
STOP = {'in','the','a','an','of','with','and','up','close','on','at','to','for','from','by','into','over','under',
        'is','are','its','their','some','very','shot','footage','video','clip','view','scene','closeup',
        'science','scientific','concept','process','diagram','illustration','animation','microscopic',
        # Style adjectives describe a look, not a subject; they must never carry a match.
        'vintage','old','modern','retro','beautiful','dramatic','cinematic','historic','historical','new','big',
        'small','large','epic','amazing','stunning','vibrant','dark','bright','slow','motion'}
# Never placed in a general-audience explainer, whatever the query.
BLOCKED = {'erotic','eroticism','nude','naked','sexy','bikini','lingerie','sensual','topless','underwear',
           'blood','gore','corpse','dead','suicide','gun','weapon','rifle','pistol','violence','drugs','cigarette',
           'alcohol','beer','vodka','whisky','smoking'}

def words(text):
    """Lowercase content words with light plural folding, so 'drops' matches 'drop'."""
    result = []
    for word in re.findall(r'[a-z]+', text.lower()):
        if word in STOP or len(word) < 2: continue
        if len(word) > 4 and word.endswith('ies'): word = word[:-3] + 'y'
        elif len(word) > 3 and word.endswith('s') and not word.endswith('ss'): word = word[:-1]
        result.append(word)
    return result

def search(query, cache=None):
    cache = cache or ROOT / '.cache/pixabay'
    cache.mkdir(parents=True, exist_ok=True)
    params = dict(q=query[:100], video_type='film', safesearch='true', per_page=20, min_width=1280)
    name = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    path = cache / (name + '.json')
    if path.exists() and time.time() - path.stat().st_mtime < 86400:
        return json.loads(path.read_text(encoding='utf-8'))['hits']
    key = os.getenv('PIXABAY_API_KEY')
    if not key:
        raise RuntimeError('PIXABAY_API_KEY is missing from .env')
    try:
        response = requests.get('https://pixabay.com/api/videos/', params={**params, 'key': key}, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f'Pixabay HTTP {response.status_code}')
        data = response.json()
    except requests.RequestException:
        raise RuntimeError('Pixabay connection failed') from None
    path.write_text(json.dumps(data), encoding='utf-8')
    return data.get('hits', [])

def rendition(hit):
    videos = hit.get('videos', {})
    for quality in ('large', 'medium', 'small'):
        video = videos.get(quality, {})
        limit = LARGE_LIMIT if quality == 'large' else MAX_BYTES
        if video.get('url') and 0 < video.get('size', 0) <= limit:
            return quality, video
    return None, None

def candidates(hits, query, used, vertical=False, need_s=0, blocked=()):
    """Relevant clips ranked best first.

    The query's first word is its subject. A clip qualifies when it carries that subject,
    at least half the query words, and one matched word among its leading tags. This
    rejects incidental subjects (a rainbow behind a balloon) and homonyms (a cell phone
    for "cell membrane"); the subject in a leading tag ranks higher.
    """
    terms = words(query)
    if not terms: return []
    head = terms[0]; terms = set(terms); blocked = set(blocked) | BLOCKED
    ranked = []
    for hit in hits:
        if hit.get('id') in used: continue
        tag_list = [t.strip() for t in hit.get('tags', '').lower().split(',') if t.strip()]
        tags = set(words(' '.join(tag_list)))
        if tags & blocked: continue
        primary = set(words(' '.join(tag_list[:3])))
        matched = terms & tags
        coverage = len(matched) / len(terms)
        # One shared word is never enough for a multi-word query ("cellular membrane" vs a phone).
        if len(matched) < min(2, len(terms)) or coverage < .5 or head not in tags or not matched & primary: continue
        quality, video = rendition(hit)
        if not video: continue
        width, height = video.get('width', 16), video.get('height', 9)
        score = 3 * coverage + (2 if head in primary else 0)
        # Matching the output orientation avoids heavy, soft crops.
        score += .8 if (height > width) == vertical else 0
        duration = hit.get('duration', 0)
        score += .6 if duration >= need_s else -.6 * min(1, (need_s - duration) / max(need_s, 1))
        score += .3 if quality == 'large' else 0
        score += min(.4, math.log10(1 + hit.get('downloads', 0)) / 15)
        ranked.append((score, hit, video))
    ranked.sort(key=lambda row: -row[0])
    return [(hit, video) for _, hit, video in ranked]

def download(url, path):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname != 'cdn.pixabay.com':
        raise ValueError('Unexpected Pixabay download host')
    temporary = path.with_suffix('.part')
    try:
        with requests.get(url, stream=True, timeout=(15, 60), allow_redirects=False) as response:
            if response.status_code != 200:
                raise RuntimeError(f'Pixabay download HTTP {response.status_code}')
            size = 0
            with temporary.open('wb') as stream:
                for chunk in response.iter_content(1024 * 256):
                    size += len(chunk)
                    if size > MAX_BYTES: raise ValueError('Stock clip exceeds 80 MB limit')
                    stream.write(chunk)
        temporary.replace(path)
    except requests.RequestException:
        raise RuntimeError('Pixabay download connection failed') from None
    finally:
        temporary.unlink(missing_ok=True)

def fetch(hit, video, target):
    """Download once into the shared cache; each run keeps its own hashed copy."""
    cached = ROOT / '.cache/pixabay/clips' / f'{hit["id"]}_{video.get("width", 0)}.mp4'
    cached.parent.mkdir(parents=True, exist_ok=True)
    if not cached.is_file(): download(video['url'], cached)
    shutil.copy2(cached, target)

def query_chain(shot):
    chain = [shot.stock_query, *getattr(shot, 'stock_alternates', [])]
    # Last resort: the first two words of the primary query.
    first = words(shot.stock_query)
    if len(first) > 2: chain.append(' '.join(first[:2]))
    seen = set(); result = []
    for query in chain:
        query = ' '.join(query.split())[:100]
        if query and query.lower() not in seen:
            seen.add(query.lower()); result.append(query)
    return result

def collect(brief, board, ctx, existing, source='auto', durations=None):
    if source == 'none': return []
    # Generic footage must never masquerade as footage of this institution.
    if brief.mode != 'general' or brief.brand == 'somaiya':
        if source == 'pixabay': raise ValueError('Pixabay requires general mode and no Somaiya branding; use approved institutional media instead.')
        return []
    if not os.getenv('PIXABAY_API_KEY'):
        if source == 'pixabay': raise RuntimeError('Set PIXABAY_API_KEY in .env')
        return []
    # Supplied media without a beat assignment is placed by the renderer; never override it.
    if any(item.get('beat_index') in (None, '') for item in existing if isinstance(item, dict)): return []
    supplied = {int(item['beat_index']) for item in existing if isinstance(item, dict)}
    vertical = brief.size[1] > brief.size[0]
    blocked = {w for phrase in brief.must_avoid for w in words(phrase) if len(w) > 3}
    records = []; used = set(); searches = 0; plan = []
    for shot in board.shots:
        if shot.beat_index in supplied: continue
        if getattr(shot, 'visual_mode', 'footage') in ('animation', 'illustration'):
            plan.append({'beat_index': shot.beat_index, 'queries_tried': [], 'chosen': None, 'fallback': shot.visual_mode + '_planned'})
            continue
        need = (durations or {}).get(shot.beat_index, 5)
        tried = []; chosen = None
        for query in query_chain(shot):
            if searches >= MAX_SEARCHES: break
            searches += 1; tried.append(query)
            try:
                choices = candidates(search(query), query, used, vertical, need, blocked)
                if not choices: continue
                hit, video = choices[0]
                path = ctx.path / 'assets' / f'pixabay_{hit["id"]}.mp4'
                fetch(hit, video, path)
            except (RuntimeError, ValueError, OSError) as exc:
                ctx.emit('asset_generator', 'stock_unavailable', inputs={'query': query, 'error_type': type(exc).__name__})
                if source == 'pixabay': raise
                continue
            used.add(hit['id'])
            # Skip a short lead-in when the clip is long enough to cover the beat.
            start = 1.0 if hit.get('duration', 0) >= need + 1.5 else 0
            chosen = dict(path=str(path), file='assets/'+path.name, source=hit['pageURL'],
                          licence='Pixabay Content License: '+LICENSE, kind='stock', sha256=sha(path),
                          beat_index=shot.beat_index, description=hit['tags'], start_s=start,
                          contributor=hit.get('user', ''), query=query, identifiable_people='review_required')
            records.append(chosen)
            ctx.emit('asset_generator', 'stock_download', inputs={'query': query, 'source': hit['pageURL']}, output_sha256=chosen['sha256'])
            break
        if not chosen and tried:
            ctx.emit('asset_generator', 'stock_no_match', inputs={'queries': tried})
        plan.append({'beat_index': shot.beat_index, 'queries_tried': tried, 'chosen': chosen['source'] if chosen else None,
                     'fallback': None if chosen else 'animated_scene'})
    ctx.artifact('stock_sources.json', records, 'asset_generator')
    ctx.artifact('stock_plan.json', plan, 'asset_generator')
    credits = ['# Footage credits', '', 'Stock footage is illustrative. Review footage for relevance, people and third-party rights before publication.', '']
    for row in records:
        credits.append(f'- Beat {row["beat_index"]}: {row["source"]} — contributor: {row["contributor"]}; {row["licence"]}')
    (ctx.path / 'CREDITS.md').write_text('\n'.join(credits), encoding='utf-8')
    return records
