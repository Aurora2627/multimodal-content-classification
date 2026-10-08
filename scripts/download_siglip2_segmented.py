"""Resume official model in bounded HTTP ranges and verify the Hub SHA256 ETag.

Run in VS Code. Existing partial weights are preserved; no remote code is executed.
"""
import hashlib, json, os, re, sys, time, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'data/cache/huggingface')
os.environ['HF_HUB_OFFLINE']='0'
os.environ['HF_HUB_DISABLE_XET']='1'
import requests
from huggingface_hub import HfApi, hf_hub_url, get_hf_file_metadata, snapshot_download
MODEL='google/siglip2-base-patch32-256'
revision="94dffa8cb1179de3e03f091dbc3917e5d5a9ae84"
url=hf_hub_url(MODEL,'model.safetensors',revision=revision)
# Pinned SHA256/size from the official Hub file metadata read during the first download.
etag='7d241bb3becad218f211f480487f491df4f8c0a472ecf7afdec5615815a301f1'
print('PINNED_MODEL',revision,flush=True)
if not re.fullmatch(r'[0-9a-f]{64}',etag or ''):
    raise RuntimeError('Official weight ETag is not SHA256')
cache=ROOT/'data/cache/huggingface/hub/models--google--siglip2-base-patch32-256'
blob=cache/'blobs'/etag;partial=blob.with_name(etag+'.incomplete')
partial.parent.mkdir(parents=True,exist_ok=True)
size=1507473288;chunk=8*1024*1024
if not blob.exists():
    while (partial.stat().st_size if partial.exists() else 0)<size:
        start=partial.stat().st_size if partial.exists() else 0
        end=min(start+chunk,size)-1
        for attempt in range(8):
            try:
                piece=partial.with_suffix('.range')
                headers=partial.with_suffix('.headers')
                process=subprocess.run(['/usr/bin/curl','--silent','--show-error','--fail',
                    '--location','--http1.1','--connect-timeout','20','--max-time','90',
                    '--range','%d-%d'%(start,end),'--dump-header',str(headers),
                    '--output',str(piece),url],capture_output=True,text=True)
                if process.returncode:
                    raise OSError('System curl transfer failed, code %d'%process.returncode)
                expected='bytes %d-%d/%d'%(start,end,size)
                ranges=re.findall(r'^content-range:\s*(.+)$',headers.read_text(),re.M|re.I)
                if not ranges or ranges[-1].strip()!=expected:
                    raise RuntimeError('Server did not honor bounded range')
                payload=piece.read_bytes()
                if len(payload)!=end-start+1:raise OSError('Truncated chunk')
                with partial.open('ab') as f:f.write(payload)
                print('WEIGHT_PROGRESS',end+1,'/',size,flush=True)
                break
            except (requests.RequestException,OSError,RuntimeError) as error:
                print('Range retry',start,attempt+1,type(error).__name__,flush=True)
                if attempt==7:raise
                time.sleep(2)
    if partial.stat().st_size!=size:raise RuntimeError('Incorrect final weight size')
    digest=hashlib.sha256()
    with partial.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=etag:
        raise RuntimeError('Weight SHA256 mismatch; partial preserved for inspection')
    partial.rename(blob)
# Small files already cached are reused. No trust_remote_code or pickle weights.
path=Path(snapshot_download(MODEL,revision=revision,
    allow_patterns=['*.json','*.txt','*.model'],local_files_only=True))
weight=path/'model.safetensors'
if not weight.exists():weight.symlink_to(Path('../../blobs')/etag)
files={}
for p in path.rglob('*'):
    if p.is_file():
        digest=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
        files[p.name]=digest.hexdigest()
if files['model.safetensors']!=etag:raise RuntimeError('Cached weight hash mismatch')
report={'model':MODEL,'revision':revision,'sha256':files,'python':sys.executable,
        'weight_size':size,'official_weight_sha256':etag,'weight_hash_verified':True}
(ROOT/'reports/siglip2-download.json').write_text(json.dumps(report,indent=2))
print('SIGLIP2_DOWNLOAD_FINISHED',revision,flush=True)
