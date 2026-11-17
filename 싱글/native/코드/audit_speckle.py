"""Read-only sensor/model audit; independent of the running optimizer."""
import argparse
import base64
import hashlib
import io
import json
import os
import re
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np


def inventory(repo,out):
    records=[];seen=set()
    for root in [repo/'싱글_하이퍼스펙/이전연구',repo/'멀티/이전연구',repo/'hyperspectral_phase',repo/'site']:
        for p in root.rglob('*'):
            if p.suffix.lower() in ('.png','.jpg','.webp') and any(x in p.name.lower() for x in ['sensor','speckle','measurement','forward']):
                records.append(dict(source=str(p),kind='saved sensor figure'))
    for html in (repo/'site').rglob('*.html'):
        s=html.read_text(encoding='utf-8')
        for i,match in enumerate(re.finditer(r'<img\b[^>]*>',s,re.I)):
            tag=match.group();src=re.search(r'src=["\']([^"\']+)',tag)
            if not src or not src[1].startswith('data:image/'):continue
            raw=base64.b64decode(src[1].split(',',1)[1]);digest=hashlib.sha256(raw).hexdigest()
            if digest in seen:continue
            seen.add(digest)
            image=Image.open(io.BytesIO(raw))
            # Keep every embedded figure large enough to be more than a thumbnail.
            if image.width<600:continue
            p=out/f'embedded_{len(seen):03}.png';image.save(p)
            records.append(dict(source=str(p),html=str(html),image_index=i,kind='embedded historical figure'))
    (out/'inventory.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    for start in range(0,len(records),12):
        sheet=Image.new('RGB',(1800,1000),'white');draw=ImageDraw.Draw(sheet)
        for k,record in enumerate(records[start:start+12]):
            p=Path(record['source']);im=Image.open(p).convert('RGB');im.thumbnail((590,215))
            x=(k%3)*600;y=(k//3)*250
            sheet.paste(im,(x,y+30));draw.text((x+5,y+5),f'{start+k}: {p.name}',fill='black')
        sheet.save(out/f'contact_{start//12:02}.jpg')
    print('Historical sensor/embedded figure inventory:',len(records),flush=True)


def forward(run,archive,out):
    import native_cli as cli
    from native_model import Model
    import torch
    from scipy import fft
    os.environ['MPLCONFIGDIR']=str(out/'mpl_cache')
    with fft.set_workers(4):
        p=np.load(run/'private_truth.npy',allow_pickle=False)
        sensor=np.load(run/'sensor.npy',allow_pickle=False)
        man=cli.loadj(run/'manifest.json')
        m=Model(archive,man['wavelength_nm'])
        field=m.field(p)
        maps={'current_saved_amplitude':sensor,'full_background_amplitude':m.cam(torch.from_numpy(np.abs(field))).numpy()}
        maps['full_background_intensity']=m.cam(torch.from_numpy(np.abs(field)**2)).numpy()
        field-=m.bg
        maps['zero_exterior_amplitude']=m.cam(torch.from_numpy(np.abs(field))).numpy()
        maps['zero_exterior_intensity']=m.cam(torch.from_numpy(np.abs(field)**2)).numpy()
        maps['background_only_amplitude']=m.cam(torch.from_numpy(np.abs(m.bg))).numpy()
        del field,m
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
    stats={}
    for ax,(name,array) in zip(axes.ravel(),maps.items()):
        shown=np.log1p(np.maximum(array[::4,::4],0)/max(float(array.mean()),1e-30))
        im=ax.imshow(shown,cmap='inferno');ax.set_title(name);fig.colorbar(im,ax=ax,shrink=.7)
        c=array[1250:1750,1250:1750]
        stats[name]=dict(shape=list(array.shape),mean=float(array.mean()),max=float(array.max()),
                         center500_energy_fraction=float(np.sum(c.astype(float)**2)/np.sum(array.astype(float)**2)))
    fig.savefig(out/'forward_comparison.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    displays=[('Posted: squared amplitude, linear',sensor.astype(float)**2,'gray'),
              ('Saved amplitude: log1p(y / mean(y))',np.log1p(sensor/max(float(sensor.mean()),1e-30)),'inferno'),
              ('Same amplitude: central 512 crop',sensor[1244:1756,1244:1756],'inferno')]
    for ax,(name,array,cmap) in zip(axes,displays):
        im=ax.imshow(array,cmap=cmap,vmin=0,vmax=np.percentile(array,99.7));ax.set_title(name);fig.colorbar(im,ax=ax,shrink=.7)
    fig.savefig(out/'display_comparison.png',dpi=130);plt.close(fig)
    stats['sensor_sha_matches']=cli.sha(sensor)==man['sensor_sha256']
    stats['noise_relative_l2']=float(np.linalg.norm((sensor-maps['full_background_amplitude']).ravel())/np.linalg.norm(sensor.ravel()))
    (out/'forward_metrics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--run',type=Path);ap.add_argument('--archive',type=Path);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    inventory(a.repo,a.output)
    if a.run:forward(a.run,a.archive,a.output)
