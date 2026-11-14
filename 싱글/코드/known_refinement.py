"""K=1 Known continuation experiment: frozen same-sensor comparison.
No truth or ROI labels enter the inverse. Do not merge as production defaults.
"""
from pathlib import Path
import argparse, gc, json, time
import numpy as np
from scipy import fft
from 공통.optics import FullB17, GEOMETRY, phase_check, digest, write_json
from 공통.object_models import N, wrap
from 싱글.코드.solver import generate, recover, evaluate, prior, sensor_check

CASES = {
    "bark": "pyramid", "sand": "pyramid", "mandrill": "pyramid",
    "barbara": "pyramid", "stream": "pyramid", "brick_wall": "pyramid",
    "random_texture": "spectral", "triple": "spectral",
}
ALPHAS=(.005,.001)
BUDGET=100
W=660.
SEED=672906
GEOM_SEED=20261127

def precondition(alpha):
    f=fft.fftfreq(N)*N; fx=f[None,:]; fy=f[:,None]
    h=2*np.interp(np.hypot(fx,fy),[0,4,16,64,128,192,224,363],
        [.876744,.876744,.455287,.415990,.299464,.068548,.057254,.057254])**2
    h+=alpha*4*(np.sin(np.pi*fx/N)**2+np.sin(np.pi*fy/N)**2)+.01
    weight=np.clip(np.median(h)/h,.125,8.);weight[0,0]=1.
    return lambda g: fft.ifft2(fft.fft2(g.reshape(N,N))*weight).real.ravel()

def refine(engine,y,w,start,dest,alpha,init_label):
    dest.mkdir(parents=True,exist_ok=False)
    y=sensor_check(y);m,cam=engine.optics(w)
    x=phase_check(start).ravel().copy()
    start_sha=digest(x.reshape(N,N))
    P=precondition(alpha);S=[];Y=[];R=[]
    counts={"forward":0,"adjoint":0};hist=[];t0=time.monotonic()
    def fg(z,grad=True):
        phase_check(z.reshape(N,N))
        counts["forward"]+=1;counts["adjoint"]+=int(grad)
        a=engine.value_grad(m,cam,z,y,0.,grad)
        e,g=prior(z,alpha,grad)
        return float(a[0]+e),(a[1].ravel()+g if grad else None),float(a[0])
    def log(k,f,d,**kw):
        row=dict(stage="LBFGS",iteration=k,objective=f,sensor_l2=float(np.sqrt(d)),
                 seconds=time.monotonic()-t0,**counts,**kw)
        hist.append(row);print(json.dumps(row),flush=True)
        write_json(dest/"history.json",hist)
    f,g,data=fg(x);log(0,f,data)
    gamma=17345.708825716574*float(g@g)/max(float(g@P(g)),1e-30)
    accepted=0;reason="iteration_budget"
    for k in range(1,BUDGET+1):
        q=g.copy();coeff=[]
        for s,v,r in zip(reversed(S),reversed(Y),reversed(R)):
            a=r*float(s@q);coeff.append(a);q-=a*v
        q=gamma*P(q)
        for s,v,r,a in zip(S,Y,R,reversed(coeff)):
            q+=s*(a-r*float(v@q))
        direction=-q;slope=float(g@direction)
        if not np.isfinite(slope) or slope>=0:
            S.clear();Y.clear();R.clear()
            direction=-gamma*P(g);slope=float(g@direction)
        if not np.isfinite(slope) or slope>=0:
            reason="non_descent_or_stationary";break
        step=min(1.,(np.pi/2)/max(float(np.max(np.abs(direction))),1e-30))
        for bt in range(35):
            s=step*direction;trial=wrap(x+s)
            nf,ng,nd=fg(trial)
            if np.isfinite(nf) and nf<=f+1e-4*step*slope:break
            step*=.5
        else:
            reason="Armijo_exhausted";break
        v=ng-g;sy=float(s@v);yy=float(v@v);ss=float(s@s)
        if sy>1e-10*np.sqrt(ss*yy) and yy>0:
            S.append(s.copy());Y.append(v.copy());R.append(1./sy)
            if len(S)>15:S.pop(0);Y.pop(0);R.pop(0)
            gamma=float(np.clip(sy/max(float(v@P(v)),1e-30),1e-3,1e8))
        x=trial;f,g,data=nf,ng,nd;accepted=k
        log(k,f,data,step=float(step),backtracks=bt,memory=len(S))
    phase=phase_check(x.reshape(N,N))
    loss,scale,pure=engine.value_grad(m,cam,x,y,0.,False)
    counts["forward"]+=1
    np.savez_compressed(dest/"recovered.npz",phase=phase,wavelength_nm=w,scale=scale)
    np.savez_compressed(dest/"predicted.npz",sensor=scale*pure)
    info=dict(wavelength_nm=w,method=f"continuation_alpha_{alpha:g}",mode="known",
        gd_iterations=0,lbfgs_iterations=accepted,completed=accepted==BUDGET,
        termination=reason,sensor_l2=float(np.sqrt(loss)),sensor_sha256=digest(y),
        phase_sha256=digest(phase),initial_phase_sha256=start_sha,initialization=init_label,
        geometry=list(GEOMETRY),B17_archive_sha256=engine.archive_sha,
        B17_source_fingerprint=engine.source_fingerprint,B17_inner_shas=engine.inner_shas,
        alpha=alpha,phase_range=[0,float(2*np.pi)],truth_in_inverse=False,
        spectral_preconditioner_source="660 nm historical sensitivity heuristic",**counts)
    write_json(dest/"recovery.json",info)
    return phase

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scene",required=True,choices=tuple(CASES))
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--random-start",action="store_true")
    a=p.parse_args()
    if a.random_start and a.scene!="random_texture":
        p.error("Random-start comparison restricted to random_texture")
    old=json.loads(a.baseline.read_text())
    old=[r for r in old if r["case"]["scene"]==a.scene]
    assert old
    expected={r["recovery"]["sensor_sha256"] for r in old}
    assert len(expected)==1
    a.output.mkdir(parents=True,exist_ok=False)
    obs=a.output/"observation"
    runs=[]
    with fft.set_workers(4):
        engine=FullB17()
        generate(engine,obs,a.scene,W,SEED,GEOM_SEED)
        with np.load(obs/"sensor_only.npz",allow_pickle=False) as z:
            y=sensor_check(z["sensor"])
        actual=digest(y)
        if actual not in expected:
            raise RuntimeError(f"Same-sensor hash mismatch: {actual} != {expected}")
        write_json(a.output/"experiment.json",dict(scene=a.scene,wavelength_nm=W,
            seed=SEED,geometry_seed=GEOM_SEED,sensor_sha256=actual,
            same_sensor_verified=True,baseline_run=43582409270,baseline_method=CASES[a.scene],
            random_start=a.random_start,alphas=list(ALPHAS),updates_per_stage=BUDGET,
            geometry=list(GEOMETRY),phase_rmse_gate_rad=.15,sensor_l2_gate_fraction=.025,
            truth_in_inverse=False,baseline_metrics=[r["evaluation"] for r in old]))
        if a.random_start:
            phase=wrap(np.random.default_rng(20261114).uniform(0,2*np.pi,(N,N)))
            init="independent_uniform_seed_20261114"
        else:
            baseline=a.output/"baseline"
            recover(engine,obs/"sensor_only.npz",baseline,"known",W,CASES[a.scene])
            with np.load(baseline/"recovered.npz",allow_pickle=False) as z:
                phase=phase_check(z["phase"]).copy()
            init=CASES[a.scene]+"_alpha_0.015"
            runs.append(baseline)
            gc.collect()
        for alpha in ALPHAS:
            stage=a.output/f"alpha_{alpha:g}"
            phase=refine(engine,y,W,phase,stage,alpha,init)
            runs.append(stage);init=stage.name;gc.collect()
        # All candidates are sealed before any truth-based evaluation.
        results={}
        for stage in runs:
            results[stage.name]=evaluate(obs/"private_truth.npz",obs/"sensor_only.npz",stage)
        write_json(a.output/"comparison.json",results)
        print("FINAL",json.dumps(results),flush=True)

if __name__=="__main__":
    main()
