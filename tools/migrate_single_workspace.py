"""One-time read-only import from the prohibited old location; no remote access."""
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'capstone_optics/싱글_하이퍼스펙'
TARGET=ROOT/'싱글/native'
AUDIT=ROOT/'실험/이관_20261117'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    records=[];excluded=SOURCE/'실험/git_result_publisher_20261117/checkout'
    for directory,dirs,files in os.walk(SOURCE):
        parent=Path(directory)
        dirs[:]=[x for x in dirs if x not in ['.git','__pycache__'] and parent/x!=excluded]
        for name in files:
            src=parent/name;relative=src.relative_to(SOURCE);dst=TARGET/relative
            if not dst.is_file():raise RuntimeError(f'Missing copied file: {relative}')
            digest=sha(src)
            if sha(dst)!=digest:raise RuntimeError(f'Copied file differs: {relative}')
            records.append(dict(path=relative.as_posix(),bytes=src.stat().st_size,sha256=digest))
    AUDIT.mkdir(parents=True,exist_ok=True)
    (AUDIT/'verified_files.json').write_text(json.dumps(dict(files=records,count=len(records),
       bytes=sum(x['bytes'] for x in records),old_git_history_copied=False,old_publisher_checkout_copied=False,
       source_preserved=True),ensure_ascii=False,indent=2),encoding='utf-8')
    print('VERIFIED',len(records),'files',sum(x['bytes'] for x in records),'bytes')
    # Reuse only the already available public project's gh-pages template.
    names=subprocess.check_output(['git','ls-tree','-r','--name-only','origin/gh-pages'],cwd=ROOT).decode().splitlines()
    site=ROOT/'site';site.mkdir(exist_ok=True)
    for name in names:
        if not (name=='index.html' or name.startswith('assets/') or name in ['asset_manifest.json','figure_provenance.json']):continue
        path=(site/name).resolve()
        if not path.is_relative_to(site.resolve()):raise ValueError('Unsafe public asset path')
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(subprocess.check_output(['git','show','origin/gh-pages:'+name],cwd=ROOT))
    s=(site/'index.html').read_text(encoding='utf-8')
    match=re.search(r'<script[^>]*id=[\'"]dataset[\'"][^>]*>(.*?)</script>',s,re.S)
    data=json.loads(match[1])
    print('DATASET_TYPE',type(data).__name__)
    sample=data.get('scenes',data) if isinstance(data,dict) else data
    print(json.dumps(sample[0] if isinstance(sample,list) else {k:v for k,v in list(sample.items())[:1]},ensure_ascii=True)[:4500])
    print('SCRIPT_TAIL',json.dumps(s[s.rfind('<script>'):][:1800],ensure_ascii=True))


if __name__=='__main__':main()
