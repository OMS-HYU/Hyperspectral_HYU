#!/usr/bin/env python3
"""Original B17 full-grid K=1 fixed-phase endpoint generalization test.

Three image families (Cameraman, Coins, Astronaut) × two wavelengths
(440, 660 nm), paired within an image with the same sensor-noise seed.
GD150 + L-BFGS150 + circular prior only; zero-phase start.
True phase kept separate from inverse, evaluated only after sealing.
Owner B17 LUT/width-map bytes are NEVER exported.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, os, time
from pathlib import Path
import numpy as np
from scipy import fft
from scipy.optimize import minimize
from skimage import data,transform,color
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

ROOT=Path.cwd() / 'b17_k1_endpoint_cloud_outputs'
REG_LAMBDA=.015
HUBER_DELTA=.7
NOBJ=512


def atomic_npz(path, **payload):
    tmp=path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(tmp,**payload)
    os.replace(tmp,path)


def penalty(x, method, lam=REG_LAMBDA, delta=HUBER_DELTA, grad=True):
    """Periodic edge prior. Quadratic branch is exactly 1-cos(Delta).

    Huber uses 2*sin(Delta/2) as a periodic chord residual: local quadratic
    rho=0.5*r*r=1-cos(Delta); local gradient=sin(Delta).
    """
    if not lam:
        return 0., np.zeros_like(x) if grad else None
    a=np.asarray(x,dtype=np.float64)
    dx=a[:,1:]-a[:,:-1];dy=a[1:,:]-a[:-1,:]
    def single(diff):
        if method=='circular':
            rho=1-np.cos(diff)
            deriv=np.sin(diff) if grad else None
        elif method=='huber070':
            r=2*np.sin(.5*diff)
            z=np.abs(r)
            rho=np.where(z<=delta,.5*r*r,delta*(z-.5*delta))
            deriv=np.clip(r,-delta,delta)*np.cos(.5*diff) if grad else None
        else:
            raise ValueError(method)
        return rho,deriv
    ax,gx=single(dx);ay,gy=single(dy)
    e=float(lam*(np.mean(ax)+np.mean(ay)))
    if not grad:return e,None
    out=np.zeros_like(a)
    fx=lam/dx.size;fy=lam/dy.size
    out[:,1:]+=fx*gx;out[:,:-1]-=fx*gx
    out[1:,:]+=fy*gy;out[:-1,:]-=fy*gy
    return e,out


def self_test():
    rng=np.random.default_rng(20261110)
    x=rng.normal(size=(11,13))*3
    direction=rng.normal(size=x.shape);direction/=np.linalg.norm(direction)
    for method in ('circular','huber070'):
        v,g=penalty(x,method)
        eps=1e-5
        fd=(penalty(x+eps*direction,method,grad=False)[0]-penalty(x-eps*direction,method,grad=False)[0])/(2*eps)
        analytic=float(np.sum(g*direction))
        err=abs(fd-analytic)/max(1e-12,abs(fd),abs(analytic))
        assert err<1e-6,(method,fd,analytic,err)
        y=x.copy();y[4,5]+=2*np.pi
        assert abs(penalty(y,method,grad=False)[0]-v)<1e-12
        print('PRIOR_FD_TEST_OK',method,'relative',err,flush=True)


SCENES=('camera','coins','astronaut')
SEEDS={'camera':261110,'coins':261111,'astronaut':261112}

def true_phase(scene):
    """Only used for generating sealed observation and for posthoc audit."""
    if scene=='camera':
        image=data.camera()
    elif scene=='coins':
        image=transform.resize(data.coins(),(512,512),order=3,
                anti_aliasing=True,preserve_range=True)
        image=np.uint8(np.rint(np.clip(image,0,255)))
    elif scene=='astronaut':
        image=np.uint8(np.rint(255*np.clip(color.rgb2gray(data.astronaut()),0,1)))
    else:raise ValueError(scene)
    assert image.shape==(512,512) and image.dtype==np.uint8
    return np.ascontiguousarray(image,dtype=np.float64)*(2*np.pi/255)


def generate(wavelength, optics, measure, destination, scene):
    obs=destination/'sensor_only.npz'
    private=destination/'private_truth.npz'
    if obs.exists() and private.exists():
        return
    if obs.exists() != private.exists():
        raise RuntimeError('incomplete cached source/truth pair')
    truth=true_phase(scene);m,cam=optics(wavelength)
    clean=measure(m,cam,truth)
    sigma=.01*float(np.mean(clean))
    rng=np.random.default_rng(SEEDS[scene])
    y=np.maximum(0.,clean+sigma*rng.normal(size=clean.shape))
    atomic_npz(obs,sensor=y,noise_sigma=np.float64(sigma),wavelength_nm=np.float64(wavelength))
    atomic_npz(private,phase=truth,wavelength_nm=np.float64(wavelength))
    print('GENERATED',scene,wavelength,'sensor_sha',sha(obs),'1% Gaussian',flush=True)
    del m,cam,clean,y,truth;gc.collect()


def fg(model,cam,x,y,value_grad,method,lam,with_grad=True):
    if with_grad:
        f,g=value_grad(model,cam,x,y,0.,True)
        if lam:
            e,ge=penalty(x,method,lam,grad=True)
            f+=e;g=g.reshape(x.shape)+ge
        else:g=g.reshape(x.shape)
        return float(f),g
    f,_,_=value_grad(model,cam,x,y,0.,False)
    if lam:f+=penalty(x,method,lam,grad=False)[0]
    return float(f)


def gradient_descent(model,cam,y,value_grad,dest,method,start,first,last):
    x=np.asarray(start,dtype=np.float64).copy();rate=0.
    reg=0. if last<=100 else REG_LAMBDA
    loss,g=fg(model,cam,x,y,value_grad,method,reg,True)
    history=[]
    for k in range(first+1,last+1):
        if k==101:
            reg=REG_LAMBDA;rate=0.
            loss,g=fg(model,cam,x,y,value_grad,method,reg,True)
        gn=float(np.sum(g*g));gmax=float(np.max(abs(g)))
        if gn<1e-42:raise RuntimeError('GD gradient vanished')
        cap=.25/max(gmax,1e-30)
        step=cap if rate==0. else min(1.5*rate,cap)
        for backs in range(32):
            cand=x-step*g
            candidate_loss=fg(model,cam,cand,y,value_grad,method,reg,False)
            if np.isfinite(candidate_loss) and candidate_loss<=loss-1e-4*step*gn:break
            step*=.5
        else:raise RuntimeError(f'Armijo failure at GD{k} {method}')
        x=cand;loss,g=fg(model,cam,x,y,value_grad,method,reg,True)
        rate=step
        if k%10==0 or k==last:
            history.append(dict(step=k,total_loss=loss,backtracks=backs))
            print('GD',method,k,'/',last,'objective',loss,flush=True)
        if k%25==0:
            atomic_npz(dest/f'gd_{method}_checkpoint.npz',phase=x,step=np.int32(k))
    return x,history


def lbfgs(model,cam,y,value_grad,dest,method,x0):
    h=[];t0=time.monotonic();evals=[0];iters=[0]
    def objective(v):
        f,g=fg(model,cam,v.reshape(NOBJ,NOBJ),y,value_grad,method,REG_LAMBDA,True)
        evals[0]+=1
        return f*1e5,(g*1e5).ravel()
    def cb(v):
        iters[0]+=1
        if iters[0]%10==0 or iters[0]==1:
            f=fg(model,cam,v.reshape(NOBJ,NOBJ),y,value_grad,method,REG_LAMBDA,False)
            h.append(dict(step=iters[0],objective=float(f),elapsed_seconds=time.monotonic()-t0))
            print('LBFGS',method,iters[0],'/150','objective',f,flush=True)
            atomic_npz(dest/f'lbfgs_{method}_checkpoint.npz',phase=v.reshape(NOBJ,NOBJ),step=np.int32(iters[0]))
    res=minimize(objective,np.asarray(x0,dtype=np.float64).ravel(),method='L-BFGS-B',jac=True,callback=cb,
                 options=dict(maxiter=150,gtol=1e-10,ftol=1e-15,maxcor=15,maxls=35))
    x=res.x.reshape(NOBJ,NOBJ)
    return x,dict(iterations=int(res.nit),function_gradient_evaluations=evals[0],termination=str(res.message),history=h)


def evaluate(wavelength,scene,method,x,observed_sha,model,cam,y,value_grad,wrapped_rmse,dest,lb_info,gd_hist):
    # Inversion is sealed BEFORE the truth file is opened by this function.
    state=dest/f'final_{method}.npz'
    if state.exists():raise FileExistsError(state)
    atomic_npz(state,phase=x,wavelength_nm=np.float64(wavelength),gd_steps=np.int32(150),lbfgs_steps=np.int32(lb_info['iterations']))
    state_hash=sha(state)
    with np.load(dest/'private_truth.npz',allow_pickle=False) as z:truth=z['phase'].copy()
    with np.load(dest/'sensor_only.npz',allow_pickle=False) as z:assert sha(dest/'sensor_only.npz')==observed_sha
    rmse,err,offset=wrapped_rmse(x,truth)
    objective,q,refit=value_grad(model,cam,x,y,0.,False)
    l2=float(np.sqrt(objective))
    result=dict(wavelength_nm=wavelength,scene=scene,prior=method,gd_steps=150,
                lbfgs_max_steps=150,lbfgs_actual_steps=lb_info['iterations'],
                phase_rmse_rad=rmse,sensor_l2=l2,
                passed=bool(rmse<=.15 and l2<=.025),
                observation_sha256=observed_sha,phase_sha256=state_hash,
                true_phase_used_in_fitting=False,known_wavelength=True,
                original_B17_physics=True,geometry=[3800,1429,512,3000],
                regularization_lambda=REG_LAMBDA,huber_delta=HUBER_DELTA if method.startswith('huber') else None,
                gd_history=gd_hist,lbfgs_info=lb_info)
    (dest/f'evaluation_{method}.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Dense 3000x3000 sensor is displayed by downsampling for figure ONLY.
    fig,axes=plt.subplots(1,5,figsize=(21,4.8),constrained_layout=True)
    im=axes[0].imshow(np.log1p(np.maximum(y[::5,::5],0)/max(float(np.mean(y)),1e-30)),cmap='inferno')
    axes[0].set_title('Observed speckle');fig.colorbar(im,ax=axes[0],shrink=.7)
    im=axes[1].imshow(np.log1p(np.maximum((q*refit)[::5,::5],0)/max(float(np.mean(y)),1e-30)),cmap='inferno')
    axes[1].set_title('Predicted speckle');fig.colorbar(im,ax=axes[1],shrink=.7)
    im=axes[2].imshow(truth,cmap='viridis',vmin=0,vmax=2*np.pi)
    axes[2].set_title('True phase');fig.colorbar(im,ax=axes[2],shrink=.7)
    im=axes[3].imshow(np.mod(x-offset,2*np.pi),cmap='viridis',vmin=0,vmax=2*np.pi)
    axes[3].set_title('Recovered phase');fig.colorbar(im,ax=axes[3],shrink=.7)
    im=axes[4].imshow(err,cmap='coolwarm',vmin=-np.pi,vmax=np.pi)
    axes[4].set_title(f'Wrapped error RMSE={rmse:.4f}');fig.colorbar(im,ax=axes[4],shrink=.7)
    for ax in axes:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(f'B17 full grid | {scene} {wavelength}nm | {method} | GD150+L-BFGS150')
    fig.savefig(dest/f'five_panels_{method}.png',dpi=110);plt.close(fig)
    print('EVALUATED',json.dumps({k:result[k] for k in ('wavelength_nm','prior','phase_rmse_rad','sensor_l2','passed','lbfgs_actual_steps')}),flush=True)
    return result


def run(wavelength,scene,method,output):
    output.mkdir(parents=True,exist_ok=True)
    from k1_cloud_batch import check_b17
    (optics,measure,value_grad,wrapped_rmse,owner_zip_sha,original_inner_hashes)=check_b17()
    audit=output/'manifest.json'
    if not audit.exists():
        audit.write_text(json.dumps({'B17_private_zip_sha256':owner_zip_sha,'B17_inner_hashes':original_inner_hashes,
            'seed':SEEDS[scene],'scene':scene,'wavelength_nm':wavelength,
            'original_grid':[3800,1429,512,3000],'phase':'2*pi*uint8/255',
            'noise':'Gaussian sigma=1% mean clean, nonnegative sensor clipping',
            'GD_1_100':'no prior, zero init','GD_101_150':'.015 circular or huber (factor matched)',
            'LBFGS':'warm GD150, maxiter=150, lambda=.015, same regularizer',
            'posthoc_only_true_phase':True},indent=2))
    generate(wavelength,optics,measure,output,scene)
    obs=output/'sensor_only.npz';observed_sha=sha(obs)
    with np.load(obs,allow_pickle=False) as z:y=z['sensor'].copy()
    with fft.set_workers(1):
        model,cam=optics(wavelength)
        common=output/'gd100_shared.npz'
        if common.exists():
            with np.load(common,allow_pickle=False) as z:base=z['phase'].copy();assert str(z['sensor_sha'])==observed_sha
        else:
            base,h0=gradient_descent(model,cam,y,value_grad,output,'circular',np.zeros((NOBJ,NOBJ)),0,100)
            atomic_npz(common,phase=base,sensor_sha=np.str_(observed_sha),step=np.int32(100))
            (output/'gd100_shared_history.json').write_text(json.dumps(h0,indent=2))
        gd150=output/f'gd150_{method}.npz'
        if gd150.exists():
            with np.load(gd150,allow_pickle=False) as z:xgd=z['phase'].copy()
            gd_hist=[]
        else:
            xgd,gd_hist=gradient_descent(model,cam,y,value_grad,output,method,base,100,150)
            atomic_npz(gd150,phase=xgd,sensor_sha=np.str_(observed_sha),step=np.int32(150))
        checkpoint=output/f'final_{method}.npz'
        if checkpoint.exists():
            print('ALREADY_COMPLETE',method,wavelength,flush=True)
            return json.loads((output/f'evaluation_{method}.json').read_text())
        x,lb_info=lbfgs(model,cam,y,value_grad,output,method,xgd)
        ans=evaluate(wavelength,scene,method,x,observed_sha,model,cam,y,value_grad,wrapped_rmse,output,lb_info,gd_hist)
        del model,cam,y,x,base,xgd;gc.collect()
    return ans


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--wavelength',type=int,choices=(440,660))
    ap.add_argument('--scene',choices=SCENES)
    ap.add_argument('--method',choices=('circular',),default='circular')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    self_test()
    if args.self_test:
        for scene in SCENES:
            x=true_phase(scene)
            assert x.shape==(512,512) and np.isfinite(x).all() and x.min()>=0 and x.max()<=2*np.pi
            print('TRUE_PHASE_SHAPE_OK',scene,flush=True)
        return
    if args.wavelength is None or args.scene is None:ap.error('wavelength and scene required')
    out=ROOT/f'{args.scene}_{args.wavelength}nm'
    result=run(args.wavelength,args.scene,args.method,out)
    print('RESULT_CLOUD_B17',json.dumps({k:result[k] for k in ('wavelength_nm','prior','phase_rmse_rad','sensor_l2','passed')}),flush=True)
if __name__=='__main__':main()