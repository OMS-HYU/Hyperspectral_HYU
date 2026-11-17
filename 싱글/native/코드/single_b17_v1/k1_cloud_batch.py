#!/usr/bin/env python3
"""Original B17 K=1 512x512 blind spectral + 100 GD + 100 circular GD cloud runner.

Each phase of the inverse reads only sensor_observation_only.npz, not private_truth.npz.
The true wavelength/phase is opened only in --stage generate or --stage evaluate.
WARNING: Reproduces frozen 2026-11-09 method, not any reduced B17 approximation.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, zipfile
from pathlib import Path
import numpy as np
from scipy import fft
from scipy.ndimage import gaussian_filter

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'cloud_k1_outputs'
DATA_ZIP=ROOT/'데이터'/'05_NN_metacam-main.zip'
LUT_SHA='02c551f2e24fb01d91416ea21cd17d24ae1c5e6fab0d630bff8e1fb2b4feaeaa'
WIDTH_SHA='b7d799c1e3a5913e7e9034fd2efc31dc2dab98822058ef08e2e57470ac2d68f9'
ORIGINAL_ZIP_SHA='6264a21cb0433c691cd4511f1f9f428f8f92b160cc4388845b6c9b0586fad716'
CASES={
 'brick_native':('brick',560,10903,False),
 'camera_rot180':('camera_rot180',610,10904,False),
 'camera_invert':('camera_invert',640,10905,False),
 'moon_flipud':('moon_flipud',530,10906,False),
 'coins_resized':('coins',590,10907,True),
 'checkerboard_resized':('checkerboard',450,10908,True),
}
def sha_bytes(a):return hashlib.sha256(a).hexdigest()
def sha(path):return sha_bytes(Path(path).read_bytes())
def check_b17():
    if not DATA_ZIP.exists():raise FileNotFoundError('B17 archive is absent from checked-out private repository')
    outer=sha(DATA_ZIP);content={}
    with zipfile.ZipFile(DATA_ZIP) as z:
        for suffix in ('bayesLUT_MSE_v6.3_nonoverlap.mat','0.6NA_random_70_1_300_1mm_mapped_width.mat'):
            f=next((n for n in z.namelist() if n.endswith(suffix)),None)
            if not f:raise RuntimeError('Missing original B17 '+suffix)
            content[suffix]=sha_bytes(z.read(f))
    if content['bayesLUT_MSE_v6.3_nonoverlap.mat']!=LUT_SHA or content['0.6NA_random_70_1_300_1mm_mapped_width.mat']!=WIDTH_SHA:
        raise RuntimeError('Cloud B17 LUT or width-map differs from owner original; REFUSE to continue: '+json.dumps(content))
    # Same exact internal physical data. Outer ZIP may differ in unrelated files/metadata.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import hyperspectral_1_b17_core as core
    core.EXPECTED_B17_SHA256=outer
    os.environ['B17_ZIP']=str(DATA_ZIP)
    from full3800_common import optics,measure,value_grad,wrapped_rmse
    return (optics,measure,value_grad,wrapped_rmse,outer,content)
def pth(name):
    p=WORK/name;p.mkdir(parents=True,exist_ok=True)
    return p
def source(name):
    from skimage import data, transform
    kind,wavelength,seed,resized=CASES[name]
    if kind=='camera_rot180':a=np.rot90(data.camera(),2)
    elif kind=='camera_invert':a=255-data.camera()
    elif kind=='moon_flipud':a=np.flipud(data.moon())
    else:a=getattr(data,kind)()
    orig=list(a.shape)
    if resized:
        a=transform.resize(a,(512,512),order=3,anti_aliasing=True,preserve_range=True)
        a=np.clip(np.rint(a),0,255).astype(np.uint8)
    if a.shape!=(512,512) or a.dtype!=np.uint8:raise ValueError('Invalid image source/geometry')
    return a,orig,wavelength,seed
def generate(name):
    (optics,measure,_,_,archivehash,contents)=check_b17()
    d=pth(name);dest=d/'sensor_observation_only.npz';truthfile=d/'private_truth.npz'
    if dest.exists() or truthfile.exists():raise FileExistsError('Refuse overwriting generated truth/sensor')
    img,orig,w,seed=source(name)
    truth=img.astype(np.float64)*(2*np.pi/255)
    t=time.monotonic();m,cam=optics(float(w))
    clean=measure(m,cam,truth)
    sigma=float(clean.mean())*.01
    rng=np.random.default_rng(seed)
    observed=np.maximum(clean+sigma*rng.normal(size=clean.shape),0)
    np.savez_compressed(dest,sensor=observed,known_channel_count=np.int32(1),noise_sigma=np.float64(sigma))
    np.savez_compressed(truthfile,true_phase_0_2pi_rad=truth,actual_wavelength_nm=np.float64(w),
                        original_shape=np.asarray(orig),original_image=img)
    meta={'case':name,'source':CASES[name][0],'original_source_shape':orig,'resized':CASES[name][3],
          'native_object_size':[512,512],'sensor_size':[3000,3000],'grid_size':[3800,3800],
          'B17_metasurface_size':[1429,1429],
          'matched_original_LUT_SHA256':LUT_SHA,'matched_original_WIDTH_SHA256':WIDTH_SHA,
          'cloud_zip_SHA256':archivehash,'original_owner_zip_SHA256':ORIGINAL_ZIP_SHA,
          'object_true_phase':'2*pi*uint8/255; stored 0...2pi','noise':'1% of mean signal',
          'observation_SHA256':sha(dest),'private_truth_SHA256':sha(truthfile),
          'elapsed_s':time.monotonic()-t}
    (d/'generation.json').write_text(json.dumps(meta,indent=2))
    print('GENERATE',name,meta,flush=True)
def scan(name):
    optics,measure,*_=check_b17()
    d=pth(name);p=d/'spectral_scan.json'
    if p.exists():raise FileExistsError(p)
    obs=d/'sensor_observation_only.npz'
    with np.load(obs,allow_pickle=False) as z:y=z['sensor'].copy()
    norm=float(np.linalg.norm(y))
    records=[]
    for w in range(440,661,10):
        t=time.monotonic()
        model,cam=optics(float(w))
        ref=measure(model,cam,np.zeros((512,512),dtype=np.float64))
        q=max(0.,float(np.sum(y*ref)/max(np.sum(ref*ref),1e-40)))
        r=float(np.linalg.norm(y-q*ref)/norm)
        records.append({'wavelength_nm':w,'relative_l2':r})
        print('SPECTRAL',name,w,'L2',round(r,7),'sec',round(time.monotonic()-t,1),flush=True)
        del model,cam,ref
    ranked=sorted(records,key=lambda x:x['relative_l2'])
    payload={'chosen_nm':ranked[0]['wavelength_nm'],'top5':ranked[:5],'records':records,
             'uses_ground_truth':False,'observation_SHA256':sha(obs),'K_known':1}
    p.write_text(json.dumps(payload,indent=2))
    print('SPECTRAL CHOSEN',name,payload['chosen_nm'],flush=True)
def camera(name):
    d=pth(name);path=d/'sensor_observation_only.npz'
    s=json.loads((d/'spectral_scan.json').read_text())
    if s['observation_SHA256']!=sha(path) or s['uses_ground_truth']:raise RuntimeError('Sensor hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        y=z['sensor'].copy();assert int(z['known_channel_count'])==1
    return d,s,float(s['chosen_nm']),y
def regularize(p,grad=True):
    dx=p[:,1:]-p[:,:-1];dy=p[1:,:]-p[:-1,:]
    E=float(np.mean(1-np.cos(dx))+np.mean(1-np.cos(dy)))
    if not grad:return E
    g=np.zeros_like(p)
    sx=np.sin(dx)/dx.size;sy=np.sin(dy)/dy.size
    g[:,1:]+=sx;g[:,:-1]-=sx;g[1:,:]+=sy;g[:-1,:]-=sy
    return E,g
def grad100(name,refinement=False):
    optics,_,value_grad,*_=check_b17()
    d,s,w,y=camera(name)
    output=d/('gd200_circular_sealed.npz' if refinement else 'gd100_sealed.npz')
    record=d/('gd200_audit.json' if refinement else 'gd100_audit.json')
    if output.exists() or record.exists():raise FileExistsError('Refuse overwriting sealed optimization')
    if refinement:
        baseline=d/'gd100_sealed.npz';before=json.loads((d/'gd100_audit.json').read_text())
        if before['accepted_steps']!=100 or before['saved_SHA256']!=sha(baseline):raise RuntimeError('Not sealed GD100')
        with np.load(baseline,allow_pickle=False) as z:x=z['phase_rad'].copy()
        cz=np.exp(1j*x)
        x=np.angle(gaussian_filter(cz.real,.65,mode='reflect')+1j*gaussian_filter(cz.imag,.65,mode='reflect'))
        prior_strength=.015;step_cap=.20
    else:
        x=np.zeros((512,512),dtype=np.float64)
        prior_strength=0.;step_cap=.25
    m,cam=optics(w);t0=time.monotonic();gradient_calls=0;forward_trials=0
    def fg(z,with_grad):
        nonlocal gradient_calls,forward_trials
        if with_grad:
            loss,g=value_grad(m,cam,z,y,0.,True);gradient_calls+=1
            if prior_strength:
                e,rg=regularize(z,True)
                return float(loss+prior_strength*e),g.reshape(z.shape)+prior_strength*rg,float(loss)
            return float(loss),g.reshape(z.shape),float(loss)
        loss,_,_=value_grad(m,cam,z,y,0.,False);forward_trials+=1
        return float(loss+prior_strength*regularize(z,False)) if prior_strength else float(loss)
    f,g,data_loss=fg(x,True);history=[{'iteration':0,'sensor_loss':data_loss,'total_loss':f}]
    rate=0.
    for i in range(1,101):
        gmax=float(np.max(abs(g)));norm=float(np.sum(g*g))
        if norm<=1e-40 or not np.isfinite(norm):raise ArithmeticError('Gradient vanished')
        cap=step_cap/max(gmax,1e-30)
        alpha=cap if rate==0 else min(rate*1.5,cap)
        for rejection in range(32):
            guess=x-alpha*g
            loss=fg(guess,False)
            if np.isfinite(loss) and loss<=f-1e-4*alpha*norm:break
            alpha*=.5
        else:raise ArithmeticError('Armijo failure '+str(i))
        x=guess;f,g,data_loss=fg(x,True);rate=alpha
        history.append({'iteration':i,'sensor_loss':data_loss,'total_loss':float(f),'rejections':rejection})
        if i%10==0 or i==1:print('GD',name,'refine' if refinement else 'base',i,'/100',data_loss,'elapsed',round(time.monotonic()-t0,1),flush=True)
        if i%25==0:np.savez_compressed(d/('refine_checkpoint.npz' if refinement else 'base_checkpoint.npz'),phase_rad=x,step=i)
    final,q,pred=value_grad(m,cam,x,y,0.,False)
    l2=float(np.linalg.norm(q*pred-y)/np.linalg.norm(y))
    np.savez_compressed(output,phase_rad=x,chosen_wavelength_nm=np.float64(w),accepted_updates=np.int32(100),sensor_l2=np.float64(l2))
    audit={'method':'circular_regularized_GD100_after_base_GD100' if refinement else 'revised_GD100_zero_init',
          'accepted_steps':100,'extra_updates':100 if refinement else 0,
          'total_accepted_updates':200 if refinement else 100,'source_wavelength_hidden_from_inverse':True,
          'GT_read_by_inverse':False,'wavelength_nm':w,'sensor_l2':l2,'sensor_loss':float(final),
          'gradient_calls':gradient_calls,'forward_only_evaluations':forward_trials+1,
          'elapsed_s':time.monotonic()-t0,'observation_SHA256':sha(d/'sensor_observation_only.npz'),
          'saved_SHA256':sha(output),'history':history}
    record.write_text(json.dumps(audit,indent=2))
    print('SEALED',name,'REFINE' if refinement else 'BASE',l2,flush=True)
def evaluate(name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    optics,measure,value_grad,wrapped_rmse,*_=check_b17()
    d=pth(name);old=json.loads((d/'gd100_audit.json').read_text());new=json.loads((d/'gd200_audit.json').read_text())
    if old['accepted_steps']!=100 or new['total_accepted_updates']!=200 or old['saved_SHA256']!=sha(d/'gd100_sealed.npz') or new['saved_SHA256']!=sha(d/'gd200_circular_sealed.npz'):raise RuntimeError('Cannot evaluate unsealed solutions')
    with np.load(d/'private_truth.npz',allow_pickle=False) as z:t=z['true_phase_0_2pi_rad'].copy();true_w=float(z['actual_wavelength_nm'])
    with np.load(d/'gd100_sealed.npz',allow_pickle=False) as z:p0=z['phase_rad'].copy()
    with np.load(d/'gd200_circular_sealed.npz',allow_pickle=False) as z:p=z['phase_rad'].copy();w=float(z['chosen_wavelength_nm'])
    _,_,_,y=camera(name)
    rm0,_,_=wrapped_rmse(p0,t);rm2,err,offset=wrapped_rmse(p,t)
    model,cam=optics(w)
    lib=measure(model,cam,np.zeros((512,512),float))
    fig,axs=plt.subplots(1,5,figsize=(25,5.4),constrained_layout=True)
    for ax,z,title in [(axs[0],y,'1. Measured speckle'),(axs[1],lib,f'2. B17 {w:g}nm library speckle (flat phase)')]:
        im=ax.imshow(np.log1p(np.maximum(z[::4,::4],0)/max(z.mean(),1e-30)),cmap='inferno')
        ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.72)
    for ax,z,title in [(axs[2],t,'3. True Phase 0–2pi'),(axs[3],np.mod(p-offset,2*np.pi),'4. Recovered Phase 0–2pi')]:
        im=ax.imshow(z,cmap='viridis',vmin=0,vmax=2*np.pi)
        ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.72,ticks=[0,np.pi,2*np.pi])
    im=axs[4].imshow(err,cmap='coolwarm',vmin=-np.pi,vmax=np.pi)
    axs[4].set_title(f'5. Wrapped Phase Error RMSE={rm2:.4f} rad')
    fig.colorbar(im,ax=axs[4],shrink=.72,ticks=[-np.pi,0,np.pi])
    for ax in axs:ax.axis('off')
    fig.suptitle(f'Original B17 synthetic K1 | {name} | GD100 + circular GD100 | 512x512 | 1% noise')
    fig.savefig(d/'five_panels.png',dpi=150);plt.close(fig)
    report={'case':name,'actual_wavelength_nm':true_w,'estimated_wavelength_nm':w,
            'wavelength_error_nm':abs(true_w-w),'GD100_phase_rmse_rad':rm0,'GD200_phase_rmse_rad':rm2,
            'GD100_sensor_l2':old['sensor_l2'],'GD200_sensor_l2':new['sensor_l2'],
            'phase_goal_pass':rm2<=.15,'sensor_goal_pass':new['sensor_l2']<=.025,
            'full_prespecified_gate_pass':w==true_w and rm2<=.15 and new['sensor_l2']<=.025,
            'phase_size':[512,512],'synthetic_only':True,'true_phase_only_posthoc':True}
    (d/'final_evaluation.json').write_text(json.dumps(report,indent=2))
    print('FINAL_CASE',json.dumps(report),flush=True)
def main():
    q=argparse.ArgumentParser()
    q.add_argument('--case',choices=CASES.keys(),required=True)
    q.add_argument('--stage',choices=['generate','scan','base','refine','evaluate','all'],default='all')
    a=q.parse_args()
    stages=['generate','scan','base','refine','evaluate'] if a.stage=='all' else [a.stage]
    with fft.set_workers(2):
        for stage in stages:
            print('START STAGE',a.case,stage,flush=True)
            {'generate':generate,'scan':scan,'base':lambda s:grad100(s,False),
             'refine':lambda s:grad100(s,True),'evaluate':evaluate}[stage](a.case)
if __name__=='__main__':main()
