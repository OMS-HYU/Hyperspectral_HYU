#!/usr/bin/env python3
"""K=1 RBC research: full-resolution GD100 + Spectral Periodic L-BFGS100."""
from __future__ import annotations
import argparse,gc,json,time
from pathlib import Path
import numpy as np
from scipy import fft
from 공통.object_models import wrap,TAU,N,RBC_MODELS,SOURCES,rbc_phase
from 공통.optics import GEOMETRY,ALPHA,GD_STEPS,LBFGS_STEPS,digest,write_json,phase_check,wavelength_check,FullB17

def prior(p,alpha=ALPHA,gradient=True):
    a=np.asarray(p,dtype=np.float64).reshape(N,N)
    dx=a[:,1:]-a[:,:-1];dy=a[1:,:]-a[:-1,:]
    e=float(alpha*(np.mean(1-np.cos(dx))+np.mean(1-np.cos(dy))))
    if not gradient:return e,None
    g=np.zeros_like(a);gx=(alpha/dx.size)*np.sin(dx);gy=(alpha/dy.size)*np.sin(dy)
    g[:,1:]+=gx;g[:,:-1]-=gx;g[1:,:]+=gy;g[:-1,:]-=gy
    return e,g.ravel()
def preconditioner(method):
    if method=="periodic":return lambda a:a.copy()
    if method!="spectral":raise ValueError(method)
    f=fft.fftfreq(N)*N;fx=f[None,:];fy=f[:,None];rad=np.hypot(fx,fy)
    knots=[0,4,16,64,128,192,224,363.]
    sens=[.876744,.876744,.455287,.415990,.299464,.068548,.057254,.057254]
    lap=4*(np.sin(np.pi*fx/N)**2+np.sin(np.pi*fy/N)**2)
    h=2*np.interp(rad,knots,sens)**2+ALPHA*lap+.01
    weight=np.clip(np.median(h)/h,.125,8.);weight[0,0]=1
    return lambda a:fft.ifft2(fft.fft2(a.reshape(N,N))*weight).real.ravel()
def sensor_check(y):
    y=np.asarray(y,dtype=np.float64)
    if y.shape!=(3000,3000) or not np.isfinite(y).all() or np.any(y<0) or not np.any(y>0):
        raise ValueError("Expected nonnegative nonzero full 3000x3000 intensity")
    return y
def profile_scale(y,p):
    return max(0.,float(np.sum(y*p)/max(np.sum(p*p),1e-40)))
def scan_wavelength(engine,y):
    """Full-grid flat-phase library scan, experimental; no truth/scene metadata."""
    norm=np.linalg.norm(y);records=[];flat=np.zeros((N,N))
    def score(w):
        p=engine.predict(w,flat);q=profile_scale(y,p)
        loss=float(np.linalg.norm(q*p-y)/norm)
        records.append({"wavelength_nm":w,"relative_l2":loss})
        print("BLIND_SCAN",w,loss,flush=True)
        del p;gc.collect()
    for w in range(440,661,10):score(w)
    best=min(records,key=lambda a:a["relative_l2"])["wavelength_nm"]
    visited={a["wavelength_nm"] for a in records}
    for w in range(max(440,best-10),min(660,best+10)+1):
        if w not in visited:score(w)
    ranked=sorted(records,key=lambda a:a["relative_l2"])
    return ranked[0]["wavelength_nm"],ranked
def fit(engine,y,w,dest,method="spectral"):
    """Exactly 100 accepted periodic GD + up to 100 periodic L-BFGS updates.
    No truth, object label, or phase-size pyramid enters this function.
    """
    y=sensor_check(y);w=wavelength_check(w)
    m,cam=engine.optics(w);x=np.zeros(N*N);hist=[];calls={"forward":0,"adjoint":0}
    t=time.monotonic()
    def fg(p,alpha,grad=True):
        phase_check(p.reshape(N,N));calls["forward"]+=1;calls["adjoint"]+=int(grad)
        out=engine.value_grad(m,cam,p,y,0.,grad);base=out[0];pr,pg=prior(p,alpha,grad)
        return base+pr,(out[1].ravel()+pg if grad else None),base
    def checkpoint(stage,k,f,data,**kw):
        row=dict(stage=stage,iteration=k,objective=float(f),sensor_l2=float(np.sqrt(data)),
            phase_min=float(x.min()),phase_max=float(x.max()),seconds=time.monotonic()-t,**calls,**kw)
        hist.append(row);print(json.dumps(row),flush=True)
        write_json(dest/"history.json",hist)
        if k in (0,25,50,75,100):
            np.savez_compressed(dest/f"{stage}_{k:03d}.npz",phase=x.reshape(N,N),iteration=k)
    f,g,data=fg(x,0);checkpoint("GD",0,f,data);step=1e5
    for k in range(1,GD_STEPS+1):
        gg=float(np.sum(g*g))
        for bt in range(25):
            trial=wrap(x-step*g);nf,_,_=fg(trial,0,False)
            if np.isfinite(nf) and nf<=f-1e-4*step*gg:break
            step*=.5
        else:raise RuntimeError(f"GD Armijo exhausted after {k-1} accepted steps")
        x=trial;f,g,data=fg(x,0);checkpoint("GD",k,f,data,step=float(step),backtracks=bt)
        step=min(step*1.3,1e7)
    np.savez_compressed(dest/"gd_sealed.npz",phase=x.reshape(N,N),iteration=100)
    P=preconditioner(method);S=[];Y=[];R=[]
    f,g,data=fg(x,ALPHA);checkpoint("LBFGS",0,f,data)
    gamma=17345.708825716574*float(g@g)/max(float(g@P(g)),1e-30)
    accepted=0;reason="iteration_budget"
    for k in range(1,LBFGS_STEPS+1):
        q=g.copy();coeff=[]
        for s,v,r in zip(reversed(S),reversed(Y),reversed(R)):
            a=r*float(s@q);coeff.append(a);q-=a*v
        q=gamma*P(q)
        for s,v,r,a in zip(S,Y,R,reversed(coeff)):q+=s*(a-r*float(v@q))
        d=-q;slope=float(g@d)
        if not np.isfinite(slope) or slope>=0:
            S.clear();Y.clear();R.clear();d=-gamma*P(g);slope=float(g@d)
        if not np.isfinite(slope) or slope>=0:
            reason="non_descent_or_stationary";break
        step=min(1.,(np.pi/2)/max(float(np.max(np.abs(d))),1e-30))
        for bt in range(35):
            s=step*d;trial=wrap(x+s);nf,ng,nd=fg(trial,ALPHA)
            if np.isfinite(nf) and nf<=f+1e-4*step*slope:break
            step*=.5
        else:
            reason="Armijo_exhausted";break
        v=ng-g;sy=float(s@v);yy=float(v@v);ss=float(s@s)
        if sy>1e-10*np.sqrt(ss*yy) and yy>0:
            S.append(s.copy());Y.append(v.copy());R.append(1/sy)
            if len(S)>15:S.pop(0);Y.pop(0);R.pop(0)
            gamma=float(np.clip(sy/max(float(v@P(v)),1e-30),1e-3,1e8))
        x=trial;f,g,data=nf,ng,nd;accepted=k
        checkpoint("LBFGS",k,f,data,step=float(step),backtracks=bt,memory=len(S))
    phase=x.reshape(N,N);phase_check(phase)
    loss,scale,pure=engine.value_grad(m,cam,x,y,0.,False);calls["forward"]+=1
    np.savez_compressed(dest/"recovered.npz",phase=phase,wavelength_nm=w,scale=scale)
    np.savez_compressed(dest/"predicted.npz",sensor=scale*pure)
    result=dict(wavelength_nm=w,method=method,gd_iterations=100,lbfgs_iterations=accepted,
        completed=accepted==100,termination=reason,sensor_l2=float(np.sqrt(loss)),
        sensor_sha256=digest(y),phase_sha256=digest(phase),geometry=list(GEOMETRY),
        B17_archive_sha256=engine.archive_sha,B17_source_fingerprint=engine.source_fingerprint,B17_inner_shas=engine.inner_shas,
        phase_range=[0,float(TAU)],alpha=ALPHA,truth_in_inverse=False,
        spectral_preconditioner_source="660 nm historical Jacobian; heuristic at other wavelengths" if method=="spectral" else "identity",
        **calls)
    write_json(dest/"recovery.json",result)
    return result
def generate(engine,dest,scene,w,seed,geometry_seed=20261127):
    dest.mkdir(parents=True,exist_ok=False);w=wavelength_check(w)
    truth=rbc_phase(scene,w,seed=geometry_seed)
    truth["phase"]=wrap(truth["phase"]);phase_check(truth["phase"])
    clean=engine.predict(w,truth["phase"]);sigma=.01*float(clean.mean())
    y=np.maximum(0.,clean+sigma*np.random.default_rng(seed).standard_normal(clean.shape))
    np.savez_compressed(dest/"sensor_only.npz",sensor=y)
    np.savez_compressed(dest/"private_truth.npz",actual_wavelength_nm=w,**truth)
    write_json(dest/"generation.json",dict(scene=scene,seed=seed,geometry_seed=geometry_seed,noise_sigma=sigma,
        noise_pct_mean=1,geometry=list(GEOMETRY),phase_range=[0,float(TAU)],
        sensor_sha256=digest(y),B17_archive_sha256=engine.archive_sha,B17_source_fingerprint=engine.source_fingerprint,
        B17_inner_shas=engine.inner_shas,sources=SOURCES,
        model_assumption="660 nm Hb index contrast held constant at other wavelengths (nondispersive approximation)"))
def recover(engine,sensor,dest,mode,w,method):
    dest.mkdir(parents=True,exist_ok=False)
    with np.load(sensor,allow_pickle=False) as a:y=sensor_check(a["sensor"])
    if mode=="blind":
        if w is not None:raise ValueError("Blind mode must not receive true wavelength")
        w,ranking=scan_wavelength(engine,y);write_json(dest/"blind_scan.json",ranking)
    elif w is None:raise ValueError("Known mode requires wavelength")
    result=fit(engine,y,w,dest,method)
    result["mode"]=mode;result["blind_experimental"]=mode=="blind"
    write_json(dest/"recovery.json",result)
    return result
def evaluate(truth_path,sensor_path,recovered_dir):
    """Independent post-hoc metrics and cyclic-colormap five-panel figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with np.load(truth_path,allow_pickle=False) as a:
        truth=phase_check(a["phase"]);w=float(a["actual_wavelength_nm"])
        labels=a["labels"].copy() if "labels" in a else None
        groups=a["groupmap"].copy() if "groupmap" in a else None
    with np.load(recovered_dir/"recovered.npz",allow_pickle=False) as a:
        p=phase_check(a["phase"]);est=float(a["wavelength_nm"])
    with np.load(sensor_path,allow_pickle=False) as a:y=sensor_check(a["sensor"])
    with np.load(recovered_dir/"predicted.npz",allow_pickle=False) as a:pred=sensor_check(a["sensor"])
    audit=json.loads((recovered_dir/"recovery.json").read_text(encoding="utf-8"))
    if digest(y)!=audit["sensor_sha256"] or digest(p)!=audit["phase_sha256"]:
        raise ValueError("Sealed sensor or recovery hash mismatch")
    delta=p-truth;off=float(np.angle(np.mean(np.exp(1j*delta))))
    err=np.angle(np.exp(1j*(delta-off)));rms=lambda a:float(np.sqrt(np.mean(a*a)))
    l2=float(np.linalg.norm(pred-y)/np.linalg.norm(y))
    metrics=dict(full_rmse_rad=rms(err),sensor_l2=l2,actual_wavelength_nm=w,
        estimated_wavelength_nm=est,wavelength_error_nm=abs(est-w),offset_rad=off,
        completed=audit["completed"],mode=audit["mode"],gd_iterations=audit["gd_iterations"],
        lbfgs_iterations=audit["lbfgs_iterations"])
    phase_pass=metrics["full_rmse_rad"]<=.15
    if labels is not None:
        metrics["roi_rmse_rad"]=rms(err[labels>0]);metrics["background_rmse_rad"]=rms(err[labels==0])
        rows=[]
        for k in (1,2,3):
            mask=labels==k
            if np.any(mask):metrics[f"layer_{k}_rmse_rad"]=rms(err[mask])
        for k in np.unique(groups[groups>0]):
            rows.append({"group":int(k),"rmse_rad":rms(err[groups==k])})
        metrics["groups"]=rows;metrics["groups_passed"]=sum(a["rmse_rad"]<=.15 for a in rows)
        metrics["groups_total"]=len(rows)
        phase_pass=phase_pass and metrics["roi_rmse_rad"]<=.15 and metrics["groups_passed"]==len(rows) and all(v<=.15 for k,v in metrics.items() if k.startswith("layer_"))
    metrics["phase_pass"]=bool(phase_pass)
    metrics["sensor_pass"]=l2<=.025
    metrics["wavelength_pass"]=abs(est-w)<=.5
    metrics["passed"]=bool(audit["completed"] and phase_pass and l2<=.025 and metrics["wavelength_pass"])
    write_json(recovered_dir/"evaluation.json",metrics)
    fig,axs=plt.subplots(1,5,figsize=(23,5),layout="constrained")
    for ax,img,title in zip(axs[:2],[y,pred],["Measured speckle","Predicted speckle"]):
        im=ax.imshow(np.log1p(img/max(y.mean(),1e-30)),cmap="inferno");ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.7)
    for ax,img,title in zip(axs[2:4],[truth,wrap(p-off)],["True phase","Recovered phase"]):
        im=ax.imshow(img,cmap="twilight",vmin=0,vmax=TAU);ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.7,ticks=[0,np.pi,TAU])
    im=axs[4].imshow(err,cmap="coolwarm",vmin=-np.pi,vmax=np.pi);axs[4].set_title(f"Circular error RMSE={metrics['full_rmse_rad']:.4f}");fig.colorbar(im,ax=axs[4],shrink=.7)
    for ax in axs:ax.axis("off")
    fig.savefig(recovered_dir/"five_panel.png",dpi=125);plt.close(fig)
    hist=json.loads((recovered_dir/"history.json").read_text())
    fig,ax=plt.subplots(figsize=(8,4));ax.semilogy([i+(100 if a["stage"]=="LBFGS" else 0) for i,a in [(v["iteration"],v) for v in hist]],[v["sensor_l2"] for v in hist]);ax.set(xlabel="Accepted update",ylabel="Sensor relative L2");fig.savefig(recovered_dir/"convergence.png",dpi=125);plt.close(fig)
    print("EVALUATED",json.dumps(metrics),flush=True)
    return metrics
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    g=sub.add_parser("generate");r=sub.add_parser("recover");e=sub.add_parser("evaluate");s=sub.add_parser("single")
    h=sub.add_parser("reference",help="Regenerate exact historical 660 nm RBC sensor and run full GD100+Spectral100")
    h.add_argument("--archive",type=Path,required=True)
    h.add_argument("--output",type=Path,required=True)
    h.add_argument("--generate-only",action="store_true")
    for a in (g,s):
        a.add_argument("--scene",choices=RBC_MODELS,default="triple")
        a.add_argument("--true-wavelength",type=float,default=660.)
        a.add_argument("--seed",type=int,default=672906)
        a.add_argument("--geometry-seed",type=int,default=20261127)
    for a in (r,s):
        a.add_argument("--mode",choices=("known","blind"),default="known")
        a.add_argument("--wavelength",type=float,help="Known inverse wavelength; forbidden for blind")
        a.add_argument("--method",choices=("spectral","periodic"),default="spectral",help="Spectral is the active RBC baseline; periodic is comparison only")
    for a in (g,r,s):a.add_argument("--output",type=Path,required=True)
    r.add_argument("--sensor",type=Path,required=True)
    e.add_argument("--truth",type=Path,required=True);e.add_argument("--sensor",type=Path,required=True);e.add_argument("--recovery",type=Path,required=True)
    a=p.parse_args(argv)
    if a.command in ("single","recover"):
        if a.mode=="blind" and a.wavelength is not None:p.error("Blind inverse must not receive wavelength")
        if a.mode=="known" and a.wavelength is None:
            if a.command=="single":a.wavelength=a.true_wavelength
            else:p.error("Known recover requires --wavelength")
    if a.command=="evaluate":evaluate(a.truth,a.sensor,a.recovery);return
    with fft.set_workers(4):
        engine=FullB17()
        if a.command=="reference":
            from 싱글.코드.fixed_reference import generate_reference,compare_reference
            a.output.mkdir(parents=True,exist_ok=False);obs=a.output/"observation";out=a.output/"inverse"
            generate_reference(engine,a.archive,obs)
            if not a.generate_only:
                recover(engine,obs/"sensor_only.npz",out,"known",660.,"spectral")
                evaluate(obs/"private_truth.npz",obs/"sensor_only.npz",out)
                compare_reference(out)
        elif a.command=="generate":generate(engine,a.output,a.scene,a.true_wavelength,a.seed,a.geometry_seed)
        elif a.command=="recover":recover(engine,a.sensor,a.output,a.mode,a.wavelength,a.method)
        else:
            a.output.mkdir(parents=True,exist_ok=False);obs=a.output/"observation";out=a.output/"inverse"
            generate(engine,obs,a.scene,a.true_wavelength,a.seed,a.geometry_seed)
            recover(engine,obs/"sensor_only.npz",out,a.mode,a.wavelength,a.method)
            evaluate(obs/"private_truth.npz",obs/"sensor_only.npz",out)
if __name__=="__main__":main()
