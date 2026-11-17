"""Full-resolution periodic RBC optimizer; no truth/ROI inputs to optimization.

Native extension: full BB/Armijo GD100, then spectral-preconditioned periodic
L-BFGS100. Positive native spectral weights are calibrated separately from
flat-phase Hessian probes, never from cell truth or ROI masks.
"""
import json
from pathlib import Path
import numpy as np
from scipy import fft
import native_cli as io

TAU=2*np.pi


def periodic(x):return np.mod(x,TAU)


def spectral_operator(weights):
    weights=np.asarray(weights,float)
    if weights.ndim!=2 or not np.isfinite(weights).all() or np.min(weights)<=0:
        raise ValueError('Positive finite Fourier weights required')
    n,m=weights.shape
    opposite=weights[np.ix_((-np.arange(n))%n,(-np.arange(m))%m)]
    if not np.allclose(weights,opposite,rtol=0,atol=1e-12):raise ValueError('Fourier weights must be even for a real self-adjoint operator')
    return lambda g:fft.ifft2(fft.fft2(g)*weights).real


def fit(out,fg,weights,identity,gd_budget=100,lbfgs_budget=100,until_updates=None,alpha=.015):
    """fg(phase,alpha)->objective,gradient,data,prior,gain. No other scene inputs."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True);P=spectral_operator(weights)
    state=out/'optimizer_state.npz';shape=weights.shape
    gd=lb=0;stage='GD';x=np.zeros(shape);xp=gp=np.empty(0);S=[];Y=[];R=[];history=[];gamma=1.
    config=dict(identity=identity,gd_budget=gd_budget,lbfgs_budget=lbfgs_budget,alpha=alpha,
                weights_sha256=io.sha(weights),shape=list(shape),full_resolution=True)
    if state.exists():
        if io.loadj(out/'config.json')!=config:raise ValueError('RBC resume configuration changed')
        with np.load(state,allow_pickle=False) as z:
            x,g,xp,gp=[z[k] for k in ['x','g','xp','gp']]
            f,data,reg,gain=z['last'];gd=int(z['gd']);lb=int(z['lb']);stage=str(z['stage']);gamma=float(z['gamma'])
            S=list(z['S']);Y=list(z['Y']);R=list(z['R']);history=json.loads(str(z['history']))
    else:
        io.write(out/'config.json',config);f,g,data,reg,gain=fg(x,0)
    termination='paused'
    def checkpoint():
        stack=lambda values:np.stack(values) if values else np.empty((0,*shape))
        io.save_state(state,x=x,g=g,xp=xp,gp=gp,last=[f,data,reg,gain],gd=gd,lb=lb,stage=stage,
                      gamma=gamma,S=stack(S),Y=stack(Y),R=np.asarray(R),history=json.dumps(history))
        io.save_phase(out/'latest_phase.npy',x)
        io.write(out/'history.json',history)
        io.write(out/'progress.json',dict(stage=stage,gd_accepted=gd,lbfgs_accepted=lb,
                 gd_target=gd_budget,lbfgs_target=lbfgs_budget,sensor_l2=float(np.sqrt(data)),
                 objective=float(f),termination=termination,phase_sha256=io.sha(x),memory=len(S),truth_used=False))
    checkpoint()
    while gd<gd_budget or lb<lbfgs_budget:
        if until_updates is not None and gd+lb>=until_updates:break
        if gd<gd_budget:
            norm=float(np.max(np.abs(g)))
            if not np.isfinite(norm) or norm<=0:termination='stationary GD';break
            step=.5/norm
            if xp.size:
                s=np.angle(np.exp(1j*(x-xp)));v=g-gp;sy=float(np.sum(s*v))
                if sy>0:step=min(step,max(float(np.sum(s*s))/sy,1e-12))
            direction=-g;slope=-float(np.sum(g*g));a=0.
        else:
            if stage=='GD':
                stage='LBFGS';f,g,data,reg,gain=fg(x,alpha);S=[];Y=[];R=[]
                gamma=.25/max(float(np.max(np.abs(P(g)))),1e-30);checkpoint()
            q=g.copy();coeff=[]
            for s,y,rho in zip(reversed(S),reversed(Y),reversed(R)):
                c=rho*float(np.sum(s*q));coeff.append(c);q-=c*y
            direction=gamma*P(q)
            for s,y,rho,c in zip(S,Y,R,reversed(coeff)):
                direction+=s*(c-rho*float(np.sum(y*direction)))
            direction=-direction;slope=float(np.sum(g*direction))
            if not np.isfinite(slope) or slope>=0:
                S=[];Y=[];R=[];direction=-gamma*P(g);slope=float(np.sum(g*direction))
            if not np.isfinite(slope) or slope>=0:termination='stationary refinement';break
            step=min(1.,(np.pi/2)/max(float(np.max(np.abs(direction))),1e-30));a=alpha
        accepted=False
        for bt in range(30):
            delta=step*direction;trial=periodic(x+delta);nf,ng,nd,nr,nq=fg(trial,a)
            if np.isfinite(nf) and np.isfinite(ng).all() and nf<=f+1e-4*step*slope:
                accepted=True;break
            step*=.5
        if not accepted:termination='Armijo exhausted';break
        if stage=='GD':xp,gp=x,g;gd+=1
        else:
            y=ng-g;sy=float(np.sum(delta*y));ss=float(np.sum(delta*delta));yy=float(np.sum(y*y))
            if sy>1e-10*np.sqrt(ss*yy) and yy>0:
                S.append(delta.copy());Y.append(y.copy());R.append(1/sy)
                if len(S)>15:S.pop(0);Y.pop(0);R.pop(0)
                gamma=float(np.clip(sy/max(float(np.sum(y*P(y))),1e-30),1e-3,1e12))
            lb+=1
        x,g,f,data,reg,gain=trial,ng,nf,nd,nr,nq
        history.append(dict(stage=stage,gd=gd,lbfgs=lb,objective=float(f),sensor_l2=float(np.sqrt(data)),backtracks=bt))
        checkpoint();print('RBC_ACCEPT',json.dumps(history[-1]),flush=True)
    if gd==gd_budget and lb==lbfgs_budget:termination='completed budgets'
    checkpoint()
    return dict(gd_accepted=gd,lbfgs_accepted=lb,termination=termination)


def calibrate(model,sensor_shape=(3000,3000),shape=(512,512),alpha=.015,seed=675999,probes=3):
    """Native, scene-independent flat-phase Hessian radial approximation."""
    import torch
    flat=np.zeros(shape);sensor=torch.from_numpy(model.measure(flat))
    freq=np.hypot(fft.fftfreq(shape[0])[:,None]*shape[0],fft.fftfreq(shape[1])[None,:]*shape[1])
    bins=np.minimum((freq/8).astype(int),45);num=np.zeros(46);den=np.zeros(46);rng=np.random.default_rng(seed)
    for k in range(probes):
        v=rng.choice([-1.,1.],size=shape);h=1e-3
        gp=model.fg(h*v,sensor,alpha)[1];gm=model.fg(-h*v,sensor,alpha)[1]
        V=fft.fft2(v);H=fft.fft2((gp-gm)/(2*h))
        num+=np.bincount(bins.ravel(),weights=np.real(np.conj(V)*H).ravel(),minlength=46)
        den+=np.bincount(bins.ravel(),weights=np.abs(V).ravel()**2,minlength=46)
        print('RBC_HESSIAN_PROBE',k+1,flush=True)
    curve=num/np.maximum(den,1e-30)
    positive=curve[curve>0]
    if positive.size==0:raise RuntimeError('Native curvature probes have no positive spectrum')
    ridge=float(np.median(positive))*.01;hdiag=np.maximum(curve[bins],ridge)+ridge
    weights=np.clip(float(np.median(hdiag))/hdiag,.125,8.);weights[0,0]=1.
    P=spectral_operator(weights);v=rng.normal(size=shape);w=rng.normal(size=shape)
    symmetry=abs(float(np.sum(v*P(w))-np.sum(P(v)*w)))/max(abs(float(np.sum(v*P(w)))),1e-30)
    if symmetry>1e-10 or float(np.sum(v*P(v)))<=0:raise RuntimeError('Spectral operator validation failed')
    return weights,dict(seed=seed,probes=probes,alpha=alpha,radial_bin_pixels=8,
                        symmetry_error=symmetry,positive=True,min_weight=float(weights.min()),
                        max_weight=float(weights.max()),truth_used=False,
                        source='native5713 flat-phase Hessian probes; no historical3800 sensitivity reuse')
