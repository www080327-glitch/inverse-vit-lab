"""Read-only reference downloads. Never load/import upstream model code or weights."""
import hashlib
import json
import urllib.request
import subprocess
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def get(url):
    # curl uses the host trust store and bounded retries for intermittent TLS EOFs.
    body=subprocess.check_output(['curl','--fail','--location','--silent','--show-error',
        '--retry','2','--retry-all-errors','--connect-timeout','10','--max-time','35',url])
    return body,url

def main():
    records=[]
    sha=json.loads(get('https://api.github.com/repos/huggingface/transformers/commits/v4.49.0')[0])['sha']
    current=json.loads(get('https://api.github.com/repos/huggingface/transformers/commits?path=src/transformers/models/qwen3_5/modeling_qwen3_5.py&per_page=1')[0])[0]['sha']
    sources=[
      ('qwen2_vl_modeling.py',f'https://raw.githubusercontent.com/huggingface/transformers/{sha}/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py'),
      ('qwen2_vl_processor.py',f'https://raw.githubusercontent.com/huggingface/transformers/{sha}/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py'),
      ('qwen3_5_modeling.py',f'https://raw.githubusercontent.com/huggingface/transformers/{current}/src/transformers/models/qwen3_5/modeling_qwen3_5.py'),
      ('Transformers_LICENSE','https://raw.githubusercontent.com/huggingface/transformers/v4.49.0/LICENSE'),
    ]
    for name,url in sources:
        body,resolved=get(url); (ROOT/'sources'/name).write_bytes(body)
        records.append(dict(file=name,url=url,resolved_url=resolved,bytes=len(body),sha256=hashlib.sha256(body).hexdigest()))
    # Papers are read on the web; full paper HTML is not republished in the code bundle.
    records.extend([dict(reference=name,url=url,kind='primary_paper_checked_in_browser') for name,url in [
        ('Qwen2-VL','https://arxiv.org/html/2409.12191v2'),
        ('ZoomEye','https://arxiv.org/html/2411.16044v1'),
        ('LLaVA-UHD','https://arxiv.org/html/2403.11703v1')]])
    # Real Earth imagery, not a generated illustration. Crop to 1920x1080 without resampling.
    url='https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57723/globe_west_2048.jpg'
    body,resolved=get(url)
    (ROOT/'data/earth_original.jpg').write_bytes(body)
    from PIL import Image
    from io import BytesIO
    image=Image.open(BytesIO(body)).convert('RGB')
    if image.width<1920 or image.height<1080: raise ValueError('NASA source unexpectedly small')
    x=(image.width-1920)//2;y=(image.height-1080)//2
    image.crop((x,y,x+1920,y+1080)).save(ROOT/'data/earth_1920x1080.png')
    records.append(dict(file='data/earth_original.jpg',url=url,resolved_url=resolved,
                        bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),original_size=list(image.size),
                        crop_xyxy=[x,y,x+1920,y+1080],interpolation=False,
                        credit='NASA Earth Observatory; Blue Marble satellite-imagery composite, 2002',
                        landing_page='https://science.nasa.gov/earth/earth-observatory/the-blue-marble-true-color-global-imagery-at-1km-resolution/'))
    (ROOT/'sources/provenance.json').write_text(json.dumps(dict(checked_utc=datetime.now(timezone.utc).isoformat(),
         qwen2_transformers_commit=sha,qwen3_5_transformers_commit=current,records=records),indent=2)+'\n')
    print('Saved pinned official code, license, source hashes, and NASA real-image crop. No model weights.')

if __name__=='__main__': main()
