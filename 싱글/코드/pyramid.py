"""Pyramid64 K=1 general-image candidate; original B17 physical grid unchanged.

Ported from historical cloud_k1_astronaut_multires / endpoint_robust.
Internal real-valued interpolation lift; optical inputs and exported phases wrapped.
"""
from __future__ import annotations
import os,time,tempfile,json
from pathlib import Path
import numpy as np
from scipy import sparse,ndimage
from scipy.optimize import minimize
from 공통.object_models import wrap,TAU,N
from 공통.optics import phase_check,wavelength_check,digest,write_json,GEOMETRY,ALPHA

SCHEDULES={"pyramid64":((64,35),(128,35),(256,40),(512,40))}
REG=REG_LAMBDA=ALPHA
NOBJ=N
HUBER_DELTA=.7

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


def make_interp(n, target=512):
    """1D linear interpolation matrix; [target x n]."""
    if not 2 <= n <= target:
        raise ValueError(n)
    if n==target:
        return sparse.identity(target, dtype=np.float64, format="csr")
    t=np.linspace(0,n-1,target)
    lo=np.floor(t).astype(np.int64)
    hi=np.minimum(lo+1,n-1)
    f=t-lo
    rows=np.concatenate([np.arange(target),np.arange(target)])
    cols=np.concatenate([lo,hi])
    weights=np.concatenate([1.-f,f])
    return sparse.coo_matrix((weights,(rows,cols)),shape=(target,n),dtype=np.float64).tocsr()


def full_from_vars(vars_, B):
    tmp=B @ np.asarray(vars_,dtype=np.float64)
    return np.asarray((B @ tmp.T).T,dtype=np.float64)


def grad_to_vars(grad_full, B):
    tmp=B.T @ np.asarray(grad_full,dtype=np.float64)
    return np.asarray((B.T @ tmp.T).T,dtype=np.float64)


def sample_full_to_vars(x, n):
    if n==512:
        return np.array(x,dtype=np.float64,copy=True)
    coords=np.linspace(0,511,n)
    yy,xx=np.meshgrid(coords,coords,indexing="ij")
    return ndimage.map_coordinates(np.asarray(x,dtype=np.float64),
                                  [yy,xx],order=1,mode="nearest")


def coupled_value_grad(model,cam,y,value_grad,phase_vars,B,step,grad=True):
    p=full_from_vars(phase_vars,B)
    lam=REG if step>100 else 0.
    if grad:
        data_loss,g=value_grad(model,cam,p,y,0.,True)
        gf=np.asarray(g).reshape(512,512)
        if lam:
            e,pg=penalty(p,"circular",lam,grad=True)
            total=float(data_loss+e)
            gf=gf+pg
        else:
            total=float(data_loss)
        return total,grad_to_vars(gf,B),float(data_loss)
    data_loss,_,_=value_grad(model,cam,p,y,0.,False)
    if lam:
        data_loss+=penalty(p,"circular",lam,grad=False)[0]
    return float(data_loss)


def gd_pyramid(model,cam,y,value_grad,dest,schedule):
    vars_=None
    prev_B=None
    history=[]
    accepted=0
    start=time.monotonic()
    for n,steps in schedule:
        B=make_interp(n)
        if vars_ is None:
            vars_=np.zeros((n,n),dtype=np.float64)
        else:
            vars_=sample_full_to_vars(full_from_vars(vars_,prev_B),n)
        rate=0.
        print("PYRAMID_STAGE_START",n,steps,"global",accepted,flush=True)
        loss,g,data=coupled_value_grad(model,cam,y,value_grad,vars_,B,accepted,True)
        for k in range(steps):
            current_step=accepted+1
            if current_step==101:
                rate=0.
                loss,g,data=coupled_value_grad(model,cam,y,value_grad,vars_,B,current_step,True)
            gn=float(np.sum(g*g))
            if gn<=1e-42 or not np.isfinite(gn):
                raise RuntimeError(f"Gradient vanished at {n=} step={current_step}")
            gmax=float(np.max(np.abs(g)))
            max_rate=.25/max(gmax,1e-30)
            rate_try=max_rate if rate==0. else min(rate*1.5,max_rate)
            for back in range(32):
                candidate=vars_-rate_try*g
                new_loss=coupled_value_grad(model,cam,y,value_grad,
                                            candidate,B,current_step,False)
                if np.isfinite(new_loss) and new_loss <= loss - 1e-4*rate_try*gn:
                    break
                rate_try*=.5
            else:
                raise RuntimeError(f"Armijo failed at stage n={n}, GD={current_step}, loss={loss}")
            vars_=candidate
            loss,g,data=coupled_value_grad(model,cam,y,value_grad,vars_,B,current_step,True)
            rate=rate_try
            accepted=current_step
            if accepted%10==0 or k+1==steps:
                rec={"gd_step":accepted,"variables_n":n,"total_obj":float(loss),
                     "sensor_loss":float(data),"backtracks":back,
                     "elapsed_s":time.monotonic()-start}
                history.append(rec)
                print("MULTIRES_GD",json.dumps(rec),flush=True)
            if accepted%25==0 or k+1==steps:
                atomic_npz(dest/f"gd_{accepted:03d}_phase.npz",
                                     phase=full_from_vars(vars_,B),
                                     gd_steps=np.int32(accepted),parameter_n=np.int32(n))
        prev_B=B
    assert accepted==150
    assert schedule[-1][0]==512
    phase=full_from_vars(vars_,prev_B)
    return phase,history


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


def fit_pyramid(engine,y,w,dest):
    """Reuse historical Pyramid64 and L-BFGS-B; wrap every optical evaluation.
    The interpolation/optimizer has a real-valued internal lift. It is NOT a
    physical signed phase or a phase-unwrapping operation. Only wrapped phase
    arrays leave the temporary working directory.
    """
    y=sensor_check(y);w=wavelength_check(w)
    from 싱글.코드.solver import sensor_check
    m,cam=engine.optics(w)
    calls={"forward":0,"adjoint":0}
    t=time.monotonic()
    def wrapped_value_grad(model,camera,x,observed,alpha,grad):
        physical=wrap(x)
        phase_check(physical.reshape(N,N))
        calls["forward"]+=1;calls["adjoint"]+=int(grad)
        return engine.value_grad(model,camera,physical,observed,alpha,grad)
    schedule=SCHEDULES["pyramid64"]
    with tempfile.TemporaryDirectory(prefix="pyramid_internal_") as temp:
        tmp=Path(temp)
        gd,gd_hist=gd_pyramid(m,cam,y,wrapped_value_grad,tmp,schedule)
        phase_check(wrap(gd))
        np.savez_compressed(dest/"gd_sealed.npz",phase=wrap(gd),iteration=150)
        final,info=lbfgs(m,cam,y,wrapped_value_grad,tmp,"circular",gd)
        # Legacy checkpoints are internal lift arrays; export wrapped copies.
        for source in sorted(tmp.glob("*.npz")):
            with np.load(source,allow_pickle=False) as a:
                payload={k:a[k] for k in a.files}
            if "phase" in payload:
                payload["phase"]=phase_check(wrap(payload["phase"]))
            np.savez_compressed(dest/source.name,**payload)
    phase=phase_check(wrap(final))
    loss,scale,pure=wrapped_value_grad(m,cam,phase,y,0.,False)
    np.savez_compressed(dest/"recovered.npz",phase=phase,wavelength_nm=w,scale=scale)
    np.savez_compressed(dest/"predicted.npz",sensor=scale*pure)
    # Legacy L-BFGS history contains objective only, not a data-only loss.
    # Do not mislabel objective as sensor L2; plot available GD points only.
    history=[dict(stage="GD",iteration=row["gd_step"],
        sensor_l2=float(np.sqrt(row["sensor_loss"])),objective=row["total_obj"],
        variables_n=row["variables_n"]) for row in gd_hist]
    history.extend(dict(stage="LBFGS",iteration=row["step"],
        objective=row["objective"]) for row in info["history"])
    history.append(dict(stage="LBFGS",iteration=info["iterations"],sensor_l2=float(np.sqrt(loss))))
    write_json(dest/"history.json",history)
    result=dict(wavelength_nm=w,method="pyramid",gd_iterations=150,
        lbfgs_iterations=info["iterations"],completed=info["iterations"]==150,
        termination=info["termination"],sensor_l2=float(np.sqrt(loss)),
        sensor_sha256=digest(y),phase_sha256=digest(phase),geometry=list(GEOMETRY),
        B17_archive_sha256=engine.archive_sha,B17_source_fingerprint=engine.source_fingerprint,B17_inner_shas=engine.inner_shas,
        phase_range=[0,float(TAU)],alpha=ALPHA,truth_in_inverse=False,
        pyramid_schedule=schedule,internal_lift=True,seconds=time.monotonic()-t,**calls)
    write_json(dest/"recovery.json",result)
    return result

