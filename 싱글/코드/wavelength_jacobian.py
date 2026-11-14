"""Controlled Cameraman K1 wavelength-adaptive preconditioning experiment.
Same camera observation and GD150 warm start, two equal 100-step periodic LBFGS runs.
Calibration never reads truth. Full native B17, gain-profiled tangent sensitivities.
"""
from pathlib import Path
import argparse, gc, json, time
import numpy as np
from scipy import fft
from 공통.optics import FullB17, GEOMETRY, ALPHA, phase_check, digest, write_json
from 공통.object_models import N, wrap
from 싱글.코드.solver import generate, evaluate, prior, sensor_check, preconditioner
from 싱글.코드.pyramid import gd_pyramid, SCHEDULES
BUDGET=100
RADII=(4,16,64,128,192,224,320)
SEED=672906

def probes(radius):
    # Distinct deterministic radial directions; exact discrete frequencies, unit phase RMS.
    t=np.arange(N,dtype=np.float64)*2*np.pi/N
    for angle in (0.,np.pi/4,np.pi/2):
        kx=int(round(radius*np.cos(angle)));ky=int(round(radius*np.sin(angle)))
        if max(abs(kx),abs(ky))>=N//2:continue
        d=np.cos(ky*t[:,None]+kx*t[None,:]+.37)
        d-=d.mean();d/=np.sqrt(np.mean(d*d))
        yield kx,ky,d

def calibrate(engine,w,phase,dest):
    """Directional projected intensity Jacobian, normalized by reference intensity L2.
    Projection removes the unobservable infinitesimal gain direction.
    The reference phase is an estimated GD phase or zero; never ground truth.
    """
    m,cam=engine.optics(w)
    u=np.exp(1j*phase.astype(np.float32)).astype(np.complex64)
    z=m.forward(u)
    ref=cam.forward(np.asarray(z.real*z.real+z.imag*z.imag,dtype=np.float64))
    rr=float(np.sum(ref*ref)); norm=np.sqrt(rr)
    assert np.isfinite(norm) and norm>0
    rows=[];calls=1
    for r in RADII:
        for kx,ky,d in probes(r):
            dz=m.forward((1j*u*d).astype(np.complex64));calls+=1
            j=cam.forward(np.asarray(2*np.real(np.conj(z)*dz),dtype=np.float64))
            j-=ref*(float(np.sum(ref*j))/rr)
            s=float(np.linalg.norm(j)/norm)
            assert np.isfinite(s) and s>0
            row=dict(radius=r,kx=kx,ky=ky,sensitivity=s)
            # Independently check one full-grid tangent via central differences.
            if r==16 and ky==0:
                eps=.002
                a=engine.measure(m,cam,wrap(phase+eps*d))
                b=engine.measure(m,cam,wrap(phase-eps*d));calls+=2
                fd=(a-b)/(2*eps);fd-=ref*(float(np.sum(ref*fd))/rr)
                error=float(np.linalg.norm(fd-j)/max(np.linalg.norm(j),1e-30))
                assert error<.03,(w,error)
                row["finite_difference_relative_error"]=error
                del a,b,fd
            rows.append(row);del dz,j
    radii=np.array(RADII,float)
    sensitivities=np.array([np.median([a["sensitivity"] for a in rows if a["radius"]==r]) for r in RADII])
    del m,cam,u,z,ref;gc.collect()
    result=dict(wavelength_nm=w,reference_phase_sha256=digest(phase),probes=rows,
        radii=radii.tolist(),sensitivities=sensitivities.tolist(),forward_calls=calls,
        normalization="gain-projected intensity tangent L2 / reference intensity L2 per unit spatial RMS phase rad")
    write_json(dest,result)
    return result

def calibrated_preconditioner(cal,alpha):
    # Same damping/clipping and Laplacian as original 660-nm heuristic.
    f=fft.fftfreq(N)*N;fx=f[None,:];fy=f[:,None]
    s=np.interp(np.hypot(fx,fy),cal["radii"],cal["sensitivities"])
    lap=4*(np.sin(np.pi*fx/N)**2+np.sin(np.pi*fy/N)**2)
    h=2*s*s+alpha*lap+.01
    weight=np.clip(np.median(h)/h,.125,8.)
    weight[0,0]=1
    assert np.isfinite(weight).all() and np.min(weight)>0
    return lambda g:fft.ifft2(fft.fft2(g.reshape(N,N))*weight).real.ravel()

def refine(engine,y,w,start,dest,alpha,init_label,P,method,source):
    dest.mkdir(parents=True,exist_ok=False)
    y=sensor_check(y);m,cam=engine.optics(w)
    x=phase_check(start).ravel().copy()
    start_sha=digest(x.reshape(N,N))
    S=[];Y=[];R=[]
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
    info=dict(wavelength_nm=w,method=method,mode="known",
        gd_iterations=150,lbfgs_iterations=accepted,completed=accepted==BUDGET,
        termination=reason,sensor_l2=float(np.sqrt(loss)),sensor_sha256=digest(y),
        phase_sha256=digest(phase),initial_phase_sha256=start_sha,initialization=init_label,
        geometry=list(GEOMETRY),B17_archive_sha256=engine.archive_sha,
        B17_source_fingerprint=engine.source_fingerprint,B17_inner_shas=engine.inner_shas,
        alpha=alpha,phase_range=[0,float(2*np.pi)],truth_in_inverse=False,
        spectral_preconditioner_source=source,**counts)
    write_json(dest/"recovery.json",info)
    return phase


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--wavelength",type=float,required=True,choices=(440.,550.,660.))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();w=a.wavelength
    a.output.mkdir(parents=True,exist_ok=False)
    obs=a.output/"observation";shared=a.output/"shared_gd";shared.mkdir()
    with fft.set_workers(4):
        engine=FullB17()
        generate(engine,obs,"camera",w,SEED,20261127)
        with np.load(obs/"sensor_only.npz",allow_pickle=False) as z:y=sensor_check(z["sensor"])
        m,cam=engine.optics(w)
        counts={"forward":0,"adjoint":0}
        def vg(model,camera,x,observed,alpha,grad):
            counts["forward"]+=1;counts["adjoint"]+=int(grad)
            return engine.value_grad(model,camera,wrap(x),observed,alpha,grad)
        phase,history=gd_pyramid(m,cam,y,vg,shared,SCHEDULES["pyramid64"])
        start=phase_check(wrap(phase));del m,cam,phase;gc.collect()
        np.savez_compressed(shared/"sealed.npz",phase=start)
        write_json(shared/"summary.json",dict(gd_iterations=150,sensor_sha256=digest(y),
            initial_phase_sha256=digest(start),history=history,**counts))
        flat=calibrate(engine,w,np.zeros((N,N)),a.output/"jacobian_flat.json")
        cal=calibrate(engine,w,start,a.output/"jacobian_gd150.json")
        # Calibration for the new preconditioner uses the shared estimated phase.
        # Flat reference is diagnostic only.
        write_json(a.output/"experiment.json",dict(scene="camera",wavelength_nm=w,seed=SEED,
          geometry=list(GEOMETRY),sensor_sha256=digest(y),initial_phase_sha256=digest(start),
          alpha=ALPHA,gd_iterations=150,lbfgs_iterations_per_arm=BUDGET,truth_in_inverse=False,
          calibration_forward_calls=flat["forward_calls"]+cal["forward_calls"],phase_rmse_gate_rad=.15,
          sensor_l2_gate_fraction=.025,comparison="fixed historical 660 nm vs wavelength-specific GD150-local projected Jacobian"))
        runs=[("fixed660",preconditioner("spectral"),"660 nm historical Jacobian"),
              ("adaptive",calibrated_preconditioner(cal,ALPHA),"current wavelength GD150-local gain-projected Jacobian")]
        for name,P,source in runs:
            refine(engine,y,w,start,a.output/name,ALPHA,"shared_Pyramid64_GD150",P,name,source)
            gc.collect()
        # Truth accessed only after both candidate solutions are sealed.
        scores={}
        for name,_,_ in runs:
            scores[name]=evaluate(obs/"private_truth.npz",obs/"sensor_only.npz",a.output/name)
        write_json(a.output/"comparison.json",scores)
        print("FINAL",json.dumps(scores),flush=True)
if __name__=="__main__":main()
