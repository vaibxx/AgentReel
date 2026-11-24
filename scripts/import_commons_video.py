"""Download a specifically selected CC0 Commons video; no API key required.

Selection stays explicit: search results are not automatically assumed to match
the script or depict Somaiya. Non-CC0 assets are rejected by this importer.
"""
import argparse,csv,hashlib,json,sys
from pathlib import Path
from urllib.parse import urlsplit
import requests
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--title',required=True); p.add_argument('--beat-index',type=int,required=True)
    p.add_argument('--output-dir',type=Path,default=ROOT/'assets/media/commons'); args=p.parse_args()
    if args.beat_index<0: raise ValueError('beat-index must be zero or greater.')
    headers={'User-Agent':'AgentReelAcademic/1.0 (local video production; explicit CC0 media import)'}
    r=requests.get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','titles':args.title,'prop':'imageinfo','iiprop':'url|size|extmetadata'},headers=headers,timeout=30); r.raise_for_status()
    page=next(iter(r.json()['query']['pages'].values())); info=page.get('imageinfo',[{}])[0]; meta=info.get('extmetadata',{})
    def field(key): return BeautifulSoup(str(meta.get(key,{}).get('value','')),'html.parser').get_text(' ',strip=True)
    if field('License').lower()!='cc0': raise ValueError('Importer accepts explicitly CC0 videos only; this asset has licence '+field('LicenseShortName'))
    url=info['url']; parts=urlsplit(url)
    if parts.scheme!='https' or parts.hostname!='upload.wikimedia.org': raise ValueError('Unexpected video host.')
    url=parts._replace(query='',fragment='').geturl()
    suffix=Path(parts.path).suffix.lower()
    if suffix not in ('.webm','.ogv','.mp4'): raise ValueError('This is not a supported video file.')
    if info.get('size',0)>80_000_000: raise ValueError('Video exceeds the 80 MB import limit.')
    folder=args.output_dir.resolve(); folder.mkdir(parents=True,exist_ok=True)
    dest=folder/(hashlib.sha256(args.title.encode()).hexdigest()[:16]+suffix)
    with requests.get(url,headers=headers,stream=True,timeout=(15,90)) as response:
        response.raise_for_status(); size=0; temporary=dest.with_suffix(suffix+'.part')
        try:
            with temporary.open('wb') as stream:
                for part in response.iter_content(1024*1024):
                    size+=len(part)
                    if size>80_000_000: raise ValueError('Video exceeds import limit.')
                    stream.write(part)
            temporary.replace(dest)
        finally:
            if temporary.exists(): temporary.unlink()
    provenance={'title':args.title,'source':info['descriptionurl'],'author':field('Artist'),'licence':'CC0 1.0','licence_url':field('LicenseUrl'),'description':field('ImageDescription'),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    dest.with_suffix('.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    manifest=folder/'commons.csv'
    rows=list(csv.DictReader(manifest.open(encoding='utf-8'))) if manifest.exists() else []
    rows=[row for row in rows if row.get('beat_index')!=str(args.beat_index)]
    rows.append({'file':dest.name,'source':provenance['source'],'licence':'CC0 1.0; '+provenance['author'],'kind':'stock','beat_index':args.beat_index,'description':provenance['description'],'start_s':0,'identifiable_people':'review_required','consent':''})
    with manifest.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[-1])); writer.writeheader(); writer.writerows(rows)
    (folder/'CREDITS.md').write_text('# Footage sources\n\n'+'\n'.join(f'- {row["source"]} — {row["licence"]}' for row in rows)+'\n\nReview the actual clip for identifiable people before use; licensing metadata does not establish consent. Stock footage is illustrative and is not evidence of Somaiya facilities.\n',encoding='utf-8')
    print('Imported:',dest); print('Manifest:',manifest)
if __name__=='__main__':
    try: main()
    except requests.HTTPError as exc:
        status=exc.response.status_code if exc.response is not None else 'unknown'
        print(f'Commons returned HTTP {status}. No usable asset was imported. Retry later or use a licensed local clip.',file=sys.stderr)
        raise SystemExit(1)
