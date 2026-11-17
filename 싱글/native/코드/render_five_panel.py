"""Five-panel post-hoc figure; never modifies optical data or reconstruction."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

TRACK=Path(__file__).resolve().parent.parent


def original_image(scene, truth):
    """Use an actual locally available source; label phase-derived fallbacks."""
    import skimage.data
    data=Path(skimage.data.data_dir)
    names={'moon':'moon.png','brick':'brick.png','chelsea':'chelsea.png',
           'camera':'camera.png','coins':'coins.png','astronaut':'astronaut.png',
           'gravel':'gravel.png','grass':'grass.png'}
    camera_transforms={'camera_rot90':lambda a:np.rot90(a), 'hflip':np.fliplr,
                       'vflip':np.flipud,'transpose':np.transpose,
                       'antitranspose':lambda a:np.flipud(np.fliplr(a.T)),
                       'rot180':lambda a:np.rot90(a,2),'rot270':lambda a:np.rot90(a,3),
                       'invert':lambda a:255-a}
    base='moon' if scene=='moon_flipud' else 'camera' if scene in camera_transforms else scene
    path=data/names[base] if base in names else None
    if scene in ('tank','boat'):
        path=TRACK/'데이터/입력이미지'/('tank.png' if scene=='tank' else 'boat.jpg')
    if path is not None and path.is_file():
        with Image.open(path) as im:array=np.asarray(im).copy()
        if scene=='moon_flipud':array=np.flipud(array)
        if scene in camera_transforms:array=camera_transforms[scene](array)
        return array,'Original image',dict(kind='local source image',path=str(path),scene_transform=scene)
    return truth/(2*np.pi),'Input map (phase-derived)',dict(kind='phase-derived grayscale; original photo unavailable',mapping='truth / (2*pi)')


def render(run, output=None, phase_path=None):
    run=Path(run)
    output=Path(output) if output else run/'recovery_float64/comparison.png'
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    # Latest 싱글스펙 conversation controls physics; the website controls figure styling only.
    if manifest.get('observation')!='amplitude' or manifest.get('geometry')!=[5713,2857,512,3000]:
        raise ValueError('Current single-spectrum native run requires 5713/2857/512/3000 amplitude data')
    truth=np.load(run/'private_truth.npy',allow_pickle=False)
    estimated=np.load(Path(phase_path) if phase_path else run/'recovery_float64/phase_060.npy',allow_pickle=False)
    amplitude=np.load(run/'sensor.npy',allow_pickle=False)
    image,title,source=original_image(manifest['scene'],truth)
    error=np.angle(np.exp(1j*(estimated-truth)))
    shown=np.log1p(np.maximum(amplitude,0)/max(float(amplitude.mean()),1e-30))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,5,figsize=(21,5.0))
    axes[0].imshow(image,cmap='gray');axes[0].set_title(title)
    speckle=axes[1].imshow(shown,cmap='inferno',vmin=0)
    axes[1].set_title('Observed speckle (amplitude)')
    fig.colorbar(speckle,ax=axes[1],fraction=.046,pad=.025,label='log(1 + A / mean(A))')
    for ax,data,label in [(axes[2],truth,'True phase'),(axes[3],np.mod(estimated,2*np.pi),'Estimated phase')]:
        im=ax.imshow(data,cmap='viridis',vmin=0,vmax=2*np.pi);ax.set_title(label)
        fig.colorbar(im,ax=ax,fraction=.046,pad=.025,label='rad',ticks=[0,np.pi,2*np.pi])
    im=axes[4].imshow(error,cmap='coolwarm',vmin=-np.pi,vmax=np.pi)
    axes[4].set_title('Circular phase error')
    fig.colorbar(im,ax=axes[4],fraction=.046,pad=.025,label='rad')
    for ax in axes:ax.set_xticks([]);ax.set_yticks([])
    policy_path=run/'completion_policy.json'
    policy=json.loads(policy_path.read_text(encoding='utf-8')) if policy_path.exists() else {}
    gd_count=policy.get('gd_accepted',150);lb_count=policy.get('lbfgs_accepted',60)
    fig.suptitle(f"{manifest['scene']} | Known {manifest['wavelength_nm']:g} nm | GD{gd_count} + float64 L{lb_count}",y=.98)
    fig.subplots_adjust(left=.015,right=.985,bottom=.12,top=.84,wspace=.32)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180,bbox_inches='tight',pad_inches=.15);plt.close(fig)
    return dict(panel_order=['original/input image','speckle intensity','true phase','estimated phase','circular phase error'],
                reference_url='https://oms-hyu.github.io/Hyperspectral_HYU/',
                physics_authority='ChatGPT 싱글스펙 latest revision; website is visual reference only',
                original_image=source,speckle=dict(source='saved native observed amplitude; no squaring or physical change',
                display='log1p(A / mean(A))',colormap='inferno',shape=list(amplitude.shape)),
                phase_limits_rad=[0,2*np.pi],phase_colormap='viridis',error_limits_rad=[-np.pi,np.pi],dpi=180)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--output',type=Path);a=ap.parse_args()
    print(json.dumps(render(a.run,a.output),ensure_ascii=False,indent=2))
