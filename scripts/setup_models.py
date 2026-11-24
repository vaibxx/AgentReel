"""Fetch free local neural voice models and openly licensed fonts."""
import hashlib,json,sys
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
FILES={
 '.cache/embeddings/model.onnx':'https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx',
 '.cache/embeddings/tokenizer.json':'https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json',
 '.cache/kokoro/model.onnx':'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx',
 '.cache/kokoro/voices.bin':'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin',
 'assets/fonts/NotoSans.ttf':'https://raw.githubusercontent.com/google/fonts/main/ofl/notosans/NotoSans%5Bwdth,wght%5D.ttf',
 'assets/fonts/NotoSansDevanagari.ttf':'https://raw.githubusercontent.com/google/fonts/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth,wght%5D.ttf',
 'assets/fonts/OFL-NotoSans.txt':'https://raw.githubusercontent.com/google/fonts/main/ofl/notosans/OFL.txt',
 'assets/fonts/OFL-NotoSansDevanagari.txt':'https://raw.githubusercontent.com/google/fonts/main/ofl/notosansdevanagari/OFL.txt'}
def main():
    manifest=[]
    for rel,url in FILES.items():
        dest=ROOT/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            print('Downloading',rel,flush=True)
            with requests.get(url,stream=True,timeout=(15,120)) as r:
                r.raise_for_status()
                temp=dest.with_suffix(dest.suffix+'.part')
                with temp.open('wb') as f:
                    for part in r.iter_content(1024*1024): f.write(part)
                temp.replace(dest)
        manifest.append({'path':rel,'url':url,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'bytes':dest.stat().st_size})
    (ROOT/'.cache/model_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Local voice and fonts ready. Model downloads are excluded from Git.')
if __name__=='__main__': main()
