"""Matrix-free variable-projection Gauss-Newton diagnostic on sealed B17 camera phase.

Exact profiled-gain Jacobian (including derivative of gain), matched adjoint,
global phase gauge removal, damped PCG and objective line search. No truth
used in inversion. Compare identical starts/observations with L-BFGS.
"""
from pathlib import Path
import argparse,gc,json,time
import numpy as np
from scipy import fft
from scipy.sparse.linalg import LinearOperator,cg
from 공통.optics import FullB17,phase_check,digest,write_json
from 공통.object_models import N,wrap
from 싱글.코드 import solver,wavelength_jacobian as jac
from 싱글.코드.noise_ablation import generate_observation
from 싱글.코드.grid_convergence import spectrum_error

W=660.;GRID=3800;M=N*N
CASES={"GN_clean_1e4":(0.,1e-4,"gn"),"GN_clean_1e6":(0.,1e-6,"gn"),
       "LBFGS_clean":(0.,0.,"lbfgs"),"GN_noise_1e4":(1.,1e-4,"gn"),
       "LBFGS_noise":(1.,0.,"lbfgs")}
OUTER=8;INNER=20

def gauge(v):
    v=np.asarray(v,dtype=np.float64).ravel()
    return v-v.mean()

class Linearization:
    def __init__(self,m,cam,x,y,counts):
        self.m=m;self.cam=cam;self.y=y;self.counts=counts
        self.u=np.exp(1j*x.reshape(N,N).astype(np.float32)).astype(np.complex64)
        self.z=m.forward(self.u);counts["forward"]+=1
        self.p=cam.forward(np.asarray(self.z.real**2+self.z.imag**2,dtype=np.float64))
        self.pp=float(np.sum(self.p*self.p))
        self.q=float(np.sum(self.p*y)/self.pp)
        assert self.q>0 and self.pp>0
        self.norm=float(np.linalg.norm(y))
        self.t=y-2*self.q*self.p
        self.r=(self.q*self.p-y)/self.norm
        self.loss=float(np.sum(self.r*self.r))
    def jvp(self,v):
        v=gauge(v).reshape(N,N)
        dz=self.m.forward((1j*self.u*v).astype(np.complex64))
        self.counts["forward"]+=1
        dp=self.cam.forward(np.asarray(2*np.real(np.conj(self.z)*dz),dtype=np.float64))
        dq=float(np.sum(self.t*dp)/self.pp)
        return (self.q*dp+self.p*dq)/self.norm
    def vjp(self,v):
        v=np.asarray(v,dtype=np.float64).reshape(self.y.shape)/self.norm
        wt=self.q*v+self.t*(float(np.sum(self.p*v))/self.pp)
        b=self.m.adjoint((self.cam.adjoint(wt)*self.z).astype(np.complex64))
        self.counts["adjoint"]+=1
        return gauge(2*np.imag(np.conj(self.u)*b))
    def check(self,x,engine):
        rng=np.random.default_rng(227)
        v=gauge(rng.standard_normal(M))
        v/=np.sqrt(np.mean(v*v))
        j=self.jvp(v)
        # Use non-random sensor test direction with nonzero Jacobian overlap.
        w=j+.1*np.linalg.norm(j)/max(np.linalg.norm(self.r),1e-30)*self.r
        a=float(np.sum(j*w));b=float(v@self.vjp(w))
        dot=abs(a-b)/max(abs(a),abs(b),1e-30)
        eps=.002
        def residual(x):
            p=engine.measure(self.m,self.cam,wrap(x.reshape(N,N)))
            self.counts["forward"]+=1
            q=float(np.sum(p*self.y)/np.sum(p*p))
            return (q*p-self.y)/self.norm
        fd=(residual(x+eps*v)-residual(x-eps*v))/(2*eps)
        fde=float(np.linalg.norm(fd-j)/max(np.linalg.norm(j),1e-30))
        assert dot<.005,(dot,a,b)
        assert fde<.03,fde
        return dict(adjoint_dot_relative_error=dot,finite_difference_relative_error=fde)

def gn(engine,y,start,dest,damping):
    dest.mkdir();m,cam=engine.optics(W)
    x=phase_check(start).ravel().copy()
    counts={"forward":0,"adjoint":0};hist=[];t0=time.monotonic()
    P=solver.preconditioner("spectral")
    lin=Linearization(m,cam,x,y,counts)
    checks=lin.check(x,engine)
    write_json(dest/"jacobian_check.json",checks)
    def log(k,**kw):
        row=dict(stage="LBFGS",iteration=k,algorithm="Gauss-Newton",
                 objective=lin.loss,sensor_l2=float(np.sqrt(lin.loss)),
                 seconds=time.monotonic()-t0,**counts,**kw)
        hist.append(row);write_json(dest/"history.json",hist);print(json.dumps(row),flush=True)
    log(0,**checks)
    accepted=0;reason="iteration_budget"
    for k in range(1,OUTER+1):
        b=-M*lin.vjp(lin.r)
        # M scales pixel-space curvature to unit-RMS phase coordinates.
        A=LinearOperator((M,M),matvec=lambda v:M*lin.vjp(lin.jvp(v))+damping*gauge(v)+v.mean(),dtype=np.float64)
        pre=LinearOperator((M,M),matvec=lambda v:gauge(P(gauge(v)))+v.mean(),dtype=np.float64)
        it=[0]
        def cb(v):it[0]+=1
        d,info=cg(A,b,M=pre,rtol=1e-3,atol=0.,maxiter=INNER,callback=cb)
        d=gauge(d);slope=-2*float(b@d)/M
        if not np.isfinite(d).all() or slope>=0:
            reason="non_descent";break
        step=min(1.,(np.pi/2)/max(float(np.max(np.abs(d))),1e-30))
        for bt in range(20):
            trial=wrap(x+step*d)
            t=Linearization(m,cam,trial,y,counts)
            if t.loss<=lin.loss+1e-4*step*slope:break
            del t;step*=.5
        else:
            reason="line_search_exhausted";break
        x=trial;lin=t;accepted=k
        log(k,cg_iterations=it[0],cg_info=int(info),step=float(step),backtracks=bt,
            damping=damping,update_rms=float(np.sqrt(np.mean((step*d)**2))))
    phase=phase_check(x.reshape(N,N))
    np.savez_compressed(dest/"recovered.npz",phase=phase,wavelength_nm=W,scale=lin.q)
    np.savez_compressed(dest/"predicted.npz",sensor=lin.q*lin.p)
    # Legacy evaluation schema: no false claim of L-BFGS updates.
    rec=dict(wavelength_nm=W,method="matrix_free_GN",mode="known",
        gd_iterations=150,lbfgs_iterations=0,gn_iterations=accepted,
        previous_lbfgs_iterations=200,completed=accepted==OUTER,termination=reason,
        sensor_l2=float(np.sqrt(lin.loss)),sensor_sha256=digest(y),
        phase_sha256=digest(phase),initial_phase_sha256=digest(start),
        geometry=list(engine.geometry),alpha=0.,damping=damping,
        truth_in_inverse=False,B17_inner_shas=engine.inner_shas,**counts)
    write_json(dest/"recovery.json",rec)
    return phase

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--case",choices=tuple(CASES),required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();noise,damping,method=CASES[a.case]
    root=a.output;root.mkdir(parents=True,exist_ok=False);obs=root/"observation"
    with fft.set_workers(4):
        engine=FullB17(propagation_n=GRID)
        gen=generate_observation(engine,obs,noise)
        gc.collect()
        with np.load(obs/"sensor_only.npz",allow_pickle=False) as z:y=solver.sensor_check(z["sensor"])
        ref=a.baseline/"inverse"
        r=json.loads((ref/"recovery.json").read_text())
        e=json.loads((ref/"evaluation.json").read_text())
        assert r["alpha"]==.005 and r["completed"]
        assert abs(e["full_rmse_rad"]-.12986880384061672)<1e-12
        with np.load(ref/"recovered.npz",allow_pickle=False) as z:start=phase_check(z["phase"])
        assert digest(start)==r["phase_sha256"]
        out=root/"inverse"
        write_json(root/"experiment.json",dict(case=a.case,noise_pct_mean=noise,geometry=list(engine.geometry),
            method=method,damping=damping,alpha=0.,initial_phase_sha256=digest(start),
            sensor_sha256=digest(y),baseline_run=43604536423,baseline_alpha=.005,
            gn_outer_budget=OUTER,gn_cg_budget=INNER,lbfgs_budget=160,truth_in_inverse=False))
        if method=="gn":
            final=gn(engine,y,start,out,damping)
        else:
            jac.BUDGET=160
            final=jac.refine(engine,y,W,start,out,0.,"sealed_alpha005",solver.preconditioner("spectral"),
                            "LBFGS_control","fixed historical spectral preconditioner")
        rec=json.loads((out/"recovery.json").read_text())
        assert rec["initial_phase_sha256"]==digest(start) and rec["sensor_sha256"]==digest(y)
        # Truth post-hoc only.
        score=solver.evaluate(obs/"private_truth.npz",obs/"sensor_only.npz",out)
        with np.load(obs/"private_truth.npz",allow_pickle=False) as z:truth=phase_check(z["phase"])
        def phase_diag(x):
            off=float(np.angle(np.mean(np.exp(1j*(x-truth)))))
            err=np.angle(np.exp(1j*(x-truth-off)))
            return dict(rmse_rad=float(np.sqrt(np.mean(err**2))),error_spectrum=spectrum_error(err))
        # Check whether the remaining actual error is a weak direction of J.
        # This is diagnostic only, computed after the result is sealed.
        m,cam=engine.optics(W)
        lin=Linearization(m,cam,final.ravel(),y,{"forward":0,"adjoint":0})
        off=float(np.angle(np.mean(np.exp(1j*(truth-final)))))
        err=np.angle(np.exp(1j*(truth-final-off))).ravel()
        tangent=lin.jvp(err)
        result=dict(case=a.case,noise_pct_mean=noise,method=method,damping=damping,
            initial=phase_diag(start),final=phase_diag(final),scores=score,recovery=rec,
            error_direction_rms=float(np.sqrt(np.mean(err**2))),
            error_direction_linearized_sensor_l2=float(np.linalg.norm(tangent)),
            same_initial_phase_sha256=digest(start),same_sensor_sha256=digest(y),
            generation=gen)
        write_json(root/"diagnostics.json",result)
        print("GN_FINAL",json.dumps(result),flush=True)

if __name__=="__main__":main()
