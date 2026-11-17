"""Use the user's exact published reference; preserve source pixels and provenance.

The example is a re-layout of published results, not a new reconstruction.
The raw-array renderer is for matching intensity/zero-exterior runs only.
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
TRACK=HERE.parent
REPO=TRACK.parent.parent
REFERENCE_URL='https://oms-hyu.github.io/Hyperspectral_HYU/'
REFERENCE_ASSET='assets/c2137cc11fa6061100d8b7e7.jpg'


def published_example(out):
    """Exact Moon 660 nm panels observed on the user's linked site."""
    from render_five_panel import original_image
    raw=subprocess.check_output(['git','show','origin/gh-pages:'+REFERENCE_ASSET],cwd=REPO)
    ref=Image.open(io.BytesIO(raw)).convert('RGB')
    if ref.size!=(1550,354):raise ValueError('Reference asset layout changed; inspect before cropping')
    original,_,source=original_image('moon',np.zeros((512,512)))
    # Entire scientific image rectangles including the original colorbars.
    # Drop predicted speckle and add the source photograph as requested.
    regions=[(2,54,310,320),(618,54,926,320),(926,54,1234,320),(1234,54,1550,320)]
    original_panel=Image.new('RGB',(308,266),'white')
    original_panel.paste(Image.fromarray(original).convert('RGB').resize((266,266)),(0,0))
    panels=[original_panel,*[ref.crop(box) for box in regions]]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,5,figsize=(22,4.8),layout='constrained')
    titles=['Original image','Observed speckle','True phase','Estimated phase','Wrapped phase error']
    for ax,panel,title in zip(axes,panels,titles):ax.imshow(panel,cmap='gray');ax.set_title(title);ax.axis('off')
    fig.suptitle('Moon | 660 nm | Published reference re-layout (GD150 + L-BFGS150)',fontsize=13)
    fig.savefig(out/'moon_660_five_panel.png',dpi=180,bbox_inches='tight');plt.close(fig)
    (out/'moon_660_reference.jpg').write_bytes(raw)
    provenance=dict(source_url=REFERENCE_URL,source_asset=REFERENCE_ASSET,
                    source_asset_sha256=hashlib.sha256(raw).hexdigest(),
                    kind='Re-layout of actual published reference, not a new reconstruction',
                    original_photo=source,source_figure_regions=regions,
                    panel_order=titles,source_geometry=[3800,1429,512,3000],
                    original_colorbars_preserved=True)
    (out/'reference_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')


def render_arrays(original,intensity,truth,estimate,output,manifest):
    """Common future layout; reject amplitude/background runs rather than relabel."""
    if manifest.get('observation')!='intensity' or manifest.get('geometry')!=[3800,1429,512,3000]:
        raise ValueError('Site reference requires 3800/1429/512/3000 intensity data; do not square amplitude observations')
    if manifest.get('amplitude_outside')!=0:
        raise ValueError('Site reference requires zero exterior; existing background-model runs are separate')
    if intensity.shape!=(3000,3000) or truth.shape!=(512,512) or estimate.shape!=truth.shape:
        raise ValueError('Unexpected reference array shape')
    if not all(np.isfinite(a).all() for a in [intensity,truth,estimate]):raise ValueError('Nonfinite figure input')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,5,figsize=(22,4.8),layout='constrained')
    axes[0].imshow(original,cmap='gray');axes[0].set_title('Original image')
    shown=np.log1p(np.maximum(intensity,0)/max(float(intensity.mean()),1e-30))
    im=axes[1].imshow(shown,cmap='inferno',vmin=0)
    axes[1].set_title('Observed speckle');fig.colorbar(im,ax=axes[1],shrink=.8,label='log(1 + I / mean(I))')
    for ax,a,title in [(axes[2],truth,'True phase'),(axes[3],np.mod(estimate,2*np.pi),'Estimated phase')]:
        im=ax.imshow(a,cmap='viridis',vmin=0,vmax=2*np.pi);ax.set_title(title)
        fig.colorbar(im,ax=ax,shrink=.8,label='rad',ticks=[0,np.pi,2*np.pi])
    error=np.angle(np.exp(1j*(estimate-truth)))
    im=axes[4].imshow(error,cmap='coolwarm',vmin=-np.pi,vmax=np.pi)
    axes[4].set_title('Wrapped phase error');fig.colorbar(im,ax=axes[4],shrink=.8,label='rad')
    for ax in axes:ax.axis('off')
    if manifest.get('figure_title'):fig.suptitle(manifest['figure_title'],fontsize=13)
    fig.savefig(output,dpi=180,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(a.output/'matplotlib_cache')
    published_example(a.output)
