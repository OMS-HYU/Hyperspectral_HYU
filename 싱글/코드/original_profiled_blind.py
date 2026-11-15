"""Profiled blind wavelength search, fixed original B17 BL=.5.
Global 10nm candidates have equal low-resolution phase-fit budgets. Retain
three separated basins, scan integer wavelengths, then 0.1nm with same
budget and common warm starts. Final full-resolution recovery sees no truth.
No physical or propagation-grid changes. All search decisions use sensor only.
"""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
from scipy import fft
from 공통.optics import phase_check,digest,write_json
from 공통.object_models import wrap,N
from 싱글.코드 import solver,pyramid,wavelength_jacobian as jac
from 싱글.코드.original_geometry import OriginalProfile
from 싱글.코드.original_blind import CASES,generate

PARAM_N=64
BUDGET=12
ALPHA=.0
def profile(engine,y,w,start=None,budget=BUDGET):
    """Equal-budget restricted phase optimization; real interpolation lift."""
    w=float(w)
    B=pyramid.make_interp(PARAM_N)
    x=np.zeros((PARAM_N,PARAM_N)) if start is None else np.asarray(start,dtype=np.float64).copy()
    m,cam=engine.optics(w)
    def fg(v,grad):
        p=wrap(pyramid.full_from_vars(v,B))
        out=engine.value_grad(m,cam,p,y,0.,grad)
        f=float(out[0])
        if not grad:return f
        g=pyramid.grad_to_vars(out[1].reshape(N,N),B)
        return f,g
    f,g=fg(x,True);initial=f;rate=0.;accepted=0;calls=1
    for k in range(budget):
        gn=float(np.sum(g*g))
        if not np.isfinite(gn) or gn<=1e-42:break
        maxrate=.25/max(float(np.max(np.abs(g))),1e-30)
        step=maxrate if rate==0 else min(rate*1.5,maxrate)
        for bt in range(24):
            trial=x-step*g;nf=fg(trial,False);calls+=1
            if np.isfinite(nf) and nf<=f-1e-4*step*gn:break
            step*=.5
        else:break
        x=trial;f,g=fg(x,True);calls+=1;rate=step;accepted+=1
    del m,cam;gc.collect()
    return dict(wavelength_nm=w,initial_l2=float(np.sqrt(initial)),profile_l2=float(np.sqrt(f)),
                accepted=accepted,budget=budget,forward_calls=calls),x

def search(engine,y,root):
    records=[];states={};cold={}
    def run(w,stage,start=None,budget=BUDGET):
        key=round(float(w),6)
        r,x=profile(engine,y,key,start,budget)
        r.update(stage=stage,initialization="flat" if start is None else "common_local_lift")
        records.append(r);states[(stage,key)]=x
        print("PROFILE_CANDIDATE",json.dumps(r),flush=True)
        write_json(root/"search_history.json",records)
        return r,x
    # Do not prefilter by flat-only residual; all 23 coarse candidates fit phase.
    for w in range(440,661,10):
        r,x=run(w,"coarse");cold[float(w)]=(r,x)
    def rank(items):return sorted(items,key=lambda a:(a[0]["profile_l2"],a[0]["wavelength_nm"]))
    coarse=rank(list(cold.values()))
    centers=[]
    for r,x in coarse:
        w=r["wavelength_nm"]
        if all(abs(w-c)>=20 for c in centers):centers.append(w)
        if len(centers)==3:break
    # Include ±10nm around each of the three distinct global basins.
    integer=sorted({float(v) for c in centers for v in range(max(440,int(c)-10),min(660,int(c)+10)+1)})
    for w in integer:
        if w not in cold:cold[w]=run(w,"integer")
    ranked=rank(list(cold.values()))
    # Refine two distinct integer minima; local candidates share identical
    # lifted phase initialization and budget within each basin.
    seeds=[]
    for r,x in ranked:
        w=r["wavelength_nm"]
        if all(abs(w-s[0]["wavelength_nm"])>=2 for s in seeds):seeds.append((r,x))
        if len(seeds)==2:break
    fine=[]
    for i,(r,x) in enumerate(seeds):
        center=r["wavelength_nm"]
        for k in range(max(4400,int(round(center*10))-10),min(6600,int(round(center*10))+10)+1):
            rr,xx=run(k/10.,f"fine{i}",x,budget=8)
            fine.append((rr,xx))
    best,low=rank(fine)[0]
    w=best["wavelength_nm"]
    write_json(root/"profile_search.json",dict(selected_wavelength_nm=w,
        coarse_centers_nm=centers,integer_seeds_nm=[r["wavelength_nm"] for r,x in seeds],
        coarse_ranking=[r for r,x in coarse],integer_ranking=[r for r,x in ranked],
        fine_ranking=[r for r,x in rank(fine)],parameter_n=PARAM_N,truth_in_inverse=False,
        sensor_sha256=digest(y),note="Fine candidates have equal local budgets; cross-basin starts differ."))
    np.savez_compressed(root/"selected_lowres.npz",phase=wrap(pyramid.full_from_vars(low,pyramid.make_interp(PARAM_N))),wavelength_nm=w)
    print("PROFILE_SELECTED",w,flush=True)
    return w

def recover(engine,sensor,root):
    with np.load(sensor,allow_pickle=False) as z:y=solver.sensor_check(z["sensor"])
    w=search(engine,y,root)
    # Full validated recovery schedule, identical to previous blind baseline.
    shared=root/"shared_gd";shared.mkdir()
    m,cam=engine.optics(w)
    def vg(model,camera,x,observed,alpha,grad):
        return engine.value_grad(model,camera,wrap(x),observed,alpha,grad)
    x,h=pyramid.gd_pyramid(m,cam,y,vg,shared,pyramid.SCHEDULES["pyramid64"])
    phase=phase_check(wrap(x));del m,cam,x;gc.collect()
    np.savez_compressed(shared/"sealed.npz",phase=phase)
    write_json(shared/"summary.json",dict(gd_iterations=150,sensor_sha256=digest(y),
        initial_phase_sha256=digest(phase),history=h))
    P=solver.preconditioner("spectral");paths=[];total=0
    for name,alpha in (("alpha015",.015),("alpha005",.005),("alpha0003",.0003)):
        out=root/name
        phase=jac.refine(engine,y,w,phase,out,alpha,"Pyramid64_GD150" if not paths else paths[-1].name+"_sealed",
                         P,"original_BL05_profiled_blind","fixed historical 660nm spectral P")
        rec=json.loads((out/"recovery.json").read_text())
        rec.update(mode="blind",blind_experimental=True,physical_profile=engine.config(),
                   selected_wavelength_nm=w,previous_lbfgs_iterations=total,
                   total_lbfgs_iterations=total+rec["lbfgs_iterations"])
        total=rec["total_lbfgs_iterations"];write_json(out/"recovery.json",rec)
        paths.append(out);gc.collect()
    return paths,w

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--case",choices=tuple(CASES),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--search-only",action="store_true")
    a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False)
    scene,true_w=CASES[a.case];obs=root/"observation"
    with fft.set_workers(4):
        engine=OriginalProfile("original660_BL05")
        write_json(root/"experiment.json",dict(physical_profile=engine.config(),case=a.case,
            truth_in_inverse=False,search_parameter_n=PARAM_N,profile_budget=BUDGET))
        generate(engine,obs,scene,true_w)
        if a.search_only:
            with np.load(obs/"sensor_only.npz",allow_pickle=False) as z:y=solver.sensor_check(z["sensor"])
            w=search(engine,y,root);scores={}
        else:
            paths,w=recover(engine,obs/"sensor_only.npz",root)
            scores={out.name:solver.evaluate(obs/"private_truth.npz",obs/"sensor_only.npz",out) for out in paths}
        # True wavelength accessed only for post-hoc comparison.
        report=json.loads((root/"profile_search.json").read_text())
        result=dict(case=a.case,scene=scene,actual_wavelength_nm=true_w,selected_wavelength_nm=w,
            wavelength_error_nm=abs(w-true_w),scores=scores,coarse_centers_nm=report["coarse_centers_nm"],
            integer_seeds_nm=report["integer_seeds_nm"],top_fine=report["fine_ranking"][:10],
            search_only=a.search_only)
        write_json(root/"comparison.json",result)
        print("PROFILED_BLIND_FINAL",json.dumps(result),flush=True)
if __name__=="__main__":main()
