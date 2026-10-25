"""Only official model and small licensed public replay data; no private inputs."""
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi,snapshot_download,hf_hub_download,get_hf_file_metadata,hf_hub_url

MODEL='Qwen/Qwen3.5-0.8B'
REVISION='2fc06364715b967f1860aea9cf38778875588b17'
TEXT_REVISION='b08601e04326c79dfdd32d625aee71d232d685c3'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def prepare_model():
    out=Path('data/p2_qwen');out.mkdir(parents=True,exist_ok=True)
    model=out/'official_model'
    if not (model/'model.safetensors-00001-of-00001.safetensors').exists():
        snapshot_download(MODEL,revision=REVISION,local_dir=model,max_workers=2,
                          allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','LICENSE','README.md'])
    manifest=dict(repo=MODEL,revision=REVISION,trust_remote_code=False,
                  files={str(p.relative_to(model)):sha(p) for p in model.rglob('*') if p.is_file() and '.cache' not in p.parts})
    filename='model.safetensors-00001-of-00001.safetensors'
    expected=get_hf_file_metadata(hf_hub_url(MODEL,filename,revision=REVISION)).etag
    if manifest['files'][filename]!=expected:raise ValueError('Official LFS weight checksum mismatch')
    manifest['official_weight_lfs_sha256']=expected
    Path('evidence/p2_qwen/model_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)


def prepare_text():
    out=Path('data/p2_qwen');out.mkdir(parents=True,exist_ok=True)
    # Download only WikiText-2 raw train and validation (< 10MB); never its test split.
    repo='Salesforce/wikitext';info=HfApi().dataset_info(repo,revision=TEXT_REVISION);rev=info.sha
    if rev!=TEXT_REVISION:raise ValueError('Pinned text revision mismatch')
    text=out/'text';text.mkdir(exist_ok=True)
    downloads=[]
    for split in ('train','validation'):
        filename=f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet'
        f=Path(hf_hub_download(repo,filename,repo_type='dataset',revision=rev,local_dir=text))
        downloads.append(dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f)))
    if sum(x['bytes'] for x in downloads)>200_000_000:raise ValueError('Public text download cap')
    readme=Path(hf_hub_download(repo,'README.md',repo_type='dataset',revision=rev,local_dir=text))
    if 'cc-by-sa-3.0' not in readme.read_text().lower():raise ValueError('Review text license before use')
    Path('evidence/p2_qwen/text_manifest.json').write_text(json.dumps(dict(repo=repo,revision=rev,license='CC-BY-SA-3.0',files=downloads,readme_sha256=sha(readme)),indent=2))


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--text-only',action='store_true');a=p.parse_args()
    if not a.text_only:prepare_model()
    prepare_text()


if __name__=='__main__':main()
