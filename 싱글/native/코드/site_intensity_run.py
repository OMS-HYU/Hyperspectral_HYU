"""Resumable single case using the user's fixed website figure/model reference."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy import fft
import scipy
import native_cli as store
from site_intensity_model import Model
from tank_precision_recovery import solve
from render_five_panel import original_image
from site_reference_figure import render_arrays

PLAN=[(64,35),(128,35),(256,40),(512,40)]
TAU=2*np.pi


def generate(a,model):
    r=a.run;mp=r/'manifest.json'
    if mp.exists():
        man=store.loadj(mp)
        if man['scene']!=a.scene or man['noise_seed']!=a.seed or man['archive_sha256']!=model.archive_sha256:
            raise ValueError('Existing run identity differs')
        return man
    if any((r/n).exists() for n in ['sensor.npy','private_truth.npy']):
        raise RuntimeError('Partial generation preserved; do not overwrite')
    truth=np.load(a.truth,allow_pickle=False)
    if truth.shape!=(512,512) or not np.isfinite(truth).all():raise ValueError('Invalid truth')
    clean=model.measure(truth)
    sigma=.01*float(clean.mean())
    sensor=np.maximum(clean+sigma*np.random.default_rng(a.seed).standard_normal(clean.shape),0)
    store.save_phase(r/'sensor.npy',sensor);store.save_phase(r/'private_truth.npy',truth)
    man=dict(scene=a.scene,wavelength_nm=660,known=True,geometry=[3800,1429,512,3000],
             amplitude_outside=0,observation='intensity',input_pitch_nm=350,sensor_pitch_um=2,
             object_meta_um=400,meta_sensor_mm=5,noise_seed=a.seed,noise_sigma=sigma,
             noise='1% mean clean intensity Gaussian, clipped nonnegative',
             sensor_sha256=store.sha(sensor),truth_sha256=store.sha(truth),
             archive_sha256=model.archive_sha256,initialization='zero; no phase transfer',
             precision='float64 propagation, original complex64 transfer coefficients',
             reference_url='https://oms-hyu.github.io/Hyperspectral_HYU/')
    store.write(mp,man);print('GENERATED',a.scene,flush=True)
    return man


def gd(run,model,sensor,identity,target=150):
    state=run/'gd_state.npz';history=[];stage=local=total=0;n=64
    x=np.zeros((n,n));xp=gp=np.empty(0)
    def lift(z,size):
        t=torch.from_numpy(z.reshape(size,size))
        return t.numpy().copy() if size==512 else F.interpolate(t[None,None],size=(512,512),mode='bilinear',align_corners=False)[0,0].numpy().copy()
    def fg(z,size):
        value,grad,data,reg,gain=model.fg(lift(z,size),sensor,0)
        if size!=512:
            t=torch.zeros((size,size),dtype=torch.float64,requires_grad=True)
            F.interpolate(t[None,None],size=(512,512),mode='bilinear',align_corners=False)[0,0].backward(torch.from_numpy(grad))
            grad=t.grad.numpy().copy()
        return value,grad,data,reg,gain
    start=time.monotonic()
    if state.exists():
        with np.load(state,allow_pickle=False) as z:
            if str(z['identity'])!=identity:raise ValueError('GD resume identity mismatch')
            x,g,xp,gp=[z[k] for k in ['x','g','xp','gp']]
            n,stage,local,total=[int(z[k]) for k in ['n','stage','local','total']]
            f,d,reg,gain=z['last'];history=json.loads(str(z['history']))
    else:f,g,d,reg,gain=fg(x,n)
    def checkpoint():
        store.save_state(state,x=x,g=g,xp=xp,gp=gp,n=n,stage=stage,local=local,total=total,
                         last=[f,d,reg,gain],history=json.dumps(history),identity=identity)
        phase=np.mod(lift(x,n),TAU);store.save_phase(run/'latest_phase.npy',phase)
        store.write(run/'progress.json',dict(gd_accepted=total,target=150,stage=stage,grid=n,
                    sensor_l2=float(np.sqrt(d)),phase_sha256=store.sha(phase)))
        store.write(run/'history.json',history)
        if total in [35,70,110,150]:store.save_phase(run/f'gd_{total:03}.npy',phase)
    checkpoint()
    while total<target:
        if local==PLAN[stage][1]:
            stage+=1;nn=PLAN[stage][0]
            x=F.interpolate(torch.from_numpy(x)[None,None],size=(nn,nn),mode='bilinear',align_corners=False)[0,0].numpy().copy()
            n=nn;local=0;xp=gp=np.empty(0);f,g,d,reg,gain=fg(x,n)
        norm=float(np.max(np.abs(g)))
        if not np.isfinite(norm) or norm<=0:raise RuntimeError('GD stationary before requested budget')
        step=.5/norm
        if xp.size:
            s=x-xp;v=g-gp;sv=float(np.sum(s*v))
            if sv>0:step=min(step,max(float(np.sum(s*s))/sv,1e-12))
        slope=float(np.sum(g*g))
        for bt in range(20):
            trial=x-step*g;nf,ng,nd,nr,nq=fg(trial,n)
            if np.isfinite(nf) and nf<=f-1e-4*step*slope:break
            step*=.5
        else:raise RuntimeError('GD Armijo failed; checkpoint preserved')
        xp,gp=x,g;x,g,f,d,reg,gain=trial,ng,nf,nd,nr,nq
        local+=1;total+=1
        history.append(dict(gd=total,grid=n,sensor_l2=float(np.sqrt(d)),objective=float(f),
                            backtracks=bt,elapsed_seconds=time.monotonic()-start))
        checkpoint();print('GD',json.dumps(history[-1]),flush=True)


def validate(model,sensor,out):
    if out.exists():
        if store.loadj(out)['relative_gradient_error']>.001:raise RuntimeError('Stored gradient validation failed')
        return
    p=.2*np.sin(np.arange(512)[None,:]/17)*np.ones((512,1))
    j,g,*_=model.fg(p,sensor,.0001)
    direction=g/max(np.max(np.abs(g)),1e-30);h=1e-3
    numeric=(model.fg(p+h*direction,sensor,.0001)[0]-model.fg(p-h*direction,sensor,.0001)[0])/(2*h)
    analytic=float(np.sum(g*direction));err=abs(numeric-analytic)/max(abs(analytic),1e-30)
    store.write(out,dict(relative_gradient_error=err,numeric=numeric,analytic=analytic,truth_used=False))
    print('GRADIENT_CHECK',err,flush=True)
    if err>.001:raise RuntimeError('Intensity gradient validation failed')


def main():
    raise RuntimeError('Cancelled physics change: 3800-grid website reference is for figure styling only; latest 싱글스펙 native protocol remains active.')
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['run','archive','truth']:ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--scene',required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--gd-until',type=int,default=150,help='Smoke/pause at a cumulative GD count; normal batch uses 150')
    a=ap.parse_args();a.run.mkdir(parents=True,exist_ok=True)
    if scipy.__version__!='1.17.0':raise ValueError('Resume requires SciPy 1.17.0')
    config={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('site_intensity_model.py')]}
    cp=a.run/'code_identity.json'
    if cp.exists() and store.loadj(cp)!=config:raise ValueError('Case code changed; use a separate run')
    store.write(cp,config)
    def stage(name):store.write(a.run/'stage.json',dict(stage=name));print('STAGE',name,flush=True)
    with fft.set_workers(4):
        stage('generate');model=Model(a.archive);man=generate(a,model)
        sensor=np.load(a.run/'sensor.npy',allow_pickle=False)
        if store.sha(sensor)!=man['sensor_sha256']:raise ValueError('Sensor hash mismatch')
        stage('validate');validate(model,sensor,a.run/'gradient_validation.json')
        if not 1<=a.gd_until<=150:raise ValueError('gd-until must be in 1..150')
        stage('gd150');gd(a.run,model,sensor,json.dumps(config,sort_keys=True),a.gd_until)
        if a.gd_until<150:
            stage('paused_smoke');return
        stage('lbfgs150')
        initial=np.load(a.run/'gd_150.npy',allow_pickle=False)
        lb=a.run/'lbfgs'
        lc=dict(**config,protocol='Website full3800 intensity; zero-exterior; GD150 + circular L-BFGS up to 150',
                sensor_sha256=man['sensor_sha256'],initial_sha256=store.sha(initial),alpha=.0001,scipy=scipy.__version__)
        count=solve(lb,initial,lambda phase:model.fg(phase,sensor,.0001),lc,150)
        phase=np.load(lb/f'phase_{count:03}.npy',allow_pickle=False) if count else initial
        store.save_phase(a.run/'final_phase.npy',phase)
        stage('evaluate')
        truth=np.load(a.run/'private_truth.npy',allow_pickle=False)
        if store.sha(truth)!=man['truth_sha256']:raise ValueError('Truth hash mismatch')
        error=np.angle(np.exp(1j*(phase-truth)))
        circular=float(np.sqrt(np.mean(error**2)))
        predicted=model.measure(phase);gain=max(0,float(np.sum(predicted*sensor)/np.sum(predicted**2)))
        l2=float(np.linalg.norm((gain*predicted-sensor).ravel())/np.linalg.norm(sensor.ravel()))
        photo,_,source=original_image(a.scene,truth)
        render_arrays(photo,sensor,truth,phase,a.run/'comparison.png',man)
        result=dict(scene=a.scene,wavelength_nm=660,geometry=man['geometry'],observation='intensity',
                    gd_accepted=150,lbfgs_accepted=count,lbfgs_requested=150,
                    termination=store.loadj(lb/'progress.json')['termination'],
                    circular_rmse_rad=circular,linear_rmse_rad=float(np.sqrt(np.mean((phase-truth)**2))),
                    sensor_l2=l2,quality_pass=circular<=.15 and l2<=.025,
                    sensor_sha256=man['sensor_sha256'],truth_sha256=man['truth_sha256'],phase_sha256=store.sha(phase),
                    original_image=source,reference_url=man['reference_url'],preview='comparison.png',
                    figure_order=['Original image','Observed speckle','True phase','Estimated phase','Wrapped phase error'],
                    phase_transfer=False,truth_used_for_evaluation_only=True)
        store.write(a.run/'evaluation.json',result);stage('completed');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
